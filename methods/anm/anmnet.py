import torch
import torch.nn as nn
from torch.autograd import Function

from modules import ClassIncrementalNetwork, SimpleLinear, backbone_dispatch
from utils.funcs import parameter_count


class HierarchicalPromptEmbedding(nn.Module):
    def __init__(self, prompt_dim, prompt_len, shared_len=2):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.task_len = prompt_len - shared_len
        self.shared_len = shared_len
        self.total_len = prompt_len

        self.shared_prompt = nn.Parameter(torch.randn(shared_len, prompt_dim) * 0.02)
        self.embedding = nn.Embedding(100, prompt_dim)  # assume up to 100 tasks
        self.generator = nn.Sequential(
            nn.Linear(prompt_dim, self.task_len * prompt_dim),
            nn.ReLU()
        )

    def forward(self, task_id, batch_size):
        task_embed = self.embedding(task_id)
        task_prompt = self.generator(task_embed).view(1, self.task_len, self.prompt_dim)
        full_prompt = torch.cat([self.shared_prompt.unsqueeze(0).expand(batch_size, -1, -1),
                                 task_prompt.expand(batch_size, -1, -1)], dim=1)
        return full_prompt  # [B, total_len, D]

class ANMNet(ClassIncrementalNetwork):
    def __init__(self, backbone_configs, network_configs, device) -> None:
        super().__init__(network_configs, device)
        self.backbone_configs = backbone_configs
        self.ta_net = backbone_dispatch(backbone_configs)

        if hasattr(self.ta_net, 'fc'):
            self.ta_net.fc = None
        self.ta_net.to(self.device)

        self.ts_nets = nn.ModuleList()
        self.classifier = None
        self.aux_classifier = None
        self.predictor = None
        

        self.prompt_dim = network_configs.get('prompt_dim', 128)
        self.prompt_len = network_configs.get('prompt_len', 5)
        self.shared_len = network_configs.get('shared_prompt_len', 2)
        self.prompt_generator = HierarchicalPromptEmbedding(self.prompt_dim, self.prompt_len, self.shared_len).to(self.device)
        self.prompt_mapper = None

    @property
    def feature_dim(self):
        return self.ts_feature_dim

    @property
    def ts_feature_dim(self):
        if len(self.ts_nets) == 0:
            return 0
        return sum(net.out_dim for net in self.ts_nets)

    @property
    def ta_feature_dim(self):
        return self.ta_net.out_dim

    @property
    def output_dim(self):
        return self.classifier.out_features if self.classifier is not None else 0

    def forward(self, x, task_id=None):
        if task_id is not None:
            task_tensor = torch.tensor([int(task_id)], device=x.device)
            prompt = self.prompt_generator(task_tensor, x.size(0))  # [B, total_len, D]
            prompt_feature = prompt.mean(dim=1)  # [B, D]

            if x.dim() == 4:
                if prompt_feature.size(1) != x.size(1):
                    if self.prompt_mapper is None:
                        self.prompt_mapper = nn.Linear(prompt_feature.size(1), x.size(1)).to(x.device)
                    prompt_feature = self.prompt_mapper(prompt_feature)
                prompt_map = prompt_feature.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
                x = x + prompt_map

        ts_outs = [net(x) for net in self.ts_nets]
        ts_features = [out['features'] for out in ts_outs]
        outputs = {'ts_features': ts_features}

        features = torch.cat(ts_features, dim=-1)
        logits = self.classifier(features)
        outputs['logits'] = logits

        if self.training:
            ta_fmap = self.ta_net(x)['fmaps'][-1]
            ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
            outputs.update({'ta_feature': ta_feature})

            if self.aux_classifier is not None:
                aux_logits = self.aux_classifier(ts_features[-1])
                outputs.update({'aux_logits': aux_logits})
            if self.predictor is not None:
                predicted_feature = self.predictor(ta_feature)
                outputs.update({'predicted_feature': predicted_feature})

        return outputs

    def update_network(self, num_new_classes, task_id=None) -> None:
        new_ts_net = backbone_dispatch(self.backbone_configs)
        if hasattr(new_ts_net, 'fc'):
            new_ts_net.fc = None
        new_ts_net.to(self.device)
        self.ts_nets.append(new_ts_net)

        if len(self.ts_nets) > 1 and (parameter_count(self.ts_nets[-1]) == parameter_count(self.ts_nets[-2])):
            if self.configs.get('init_from_last'):
                self.ts_nets[-1].load_state_dict(self.ts_nets[-2].state_dict())
            elif self.configs.get('init_from_interpolation'):
                gamma = self.configs['init_interpolation_factor']
                for p_ta, p_ts_old, p_ts_new in zip(
                    self.ta_net.parameters(),
                    self.ts_nets[-2].parameters(),
                    self.ts_nets[-1].parameters()
                ):
                    p_ts_new.data = gamma * p_ts_old.data + (1 - gamma) * p_ts_new.data

        new_dim = new_ts_net.out_dim
        classifier = SimpleLinear(self.feature_dim, self.output_dim + num_new_classes, device=self.device)
        if self.classifier is not None:
            classifier.weight.data[:self.output_dim, :-new_dim] = self.classifier.weight.data
            classifier.weight.data[:self.output_dim, -new_dim:] = 0.
            classifier.bias.data[:self.output_dim] = self.classifier.bias.data
        self.classifier = classifier

        if len(self.ts_nets) > 1:
            self.aux_classifier = SimpleLinear(new_dim, num_new_classes + 1, device=self.device)
            if self.predictor is None:
                self.predictor = SimpleLinear(self.ta_feature_dim, self.ta_feature_dim, device=self.device)
                

    def weight_align(self, num_new_classes):
        new = self.classifier.weight.data[-num_new_classes:].norm(dim=-1).mean()
        old = self.classifier.weight.data[:-num_new_classes].norm(dim=-1).mean()
        self.classifier.weight.data[-num_new_classes:] *= old / new

    def train(self, mode: bool=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for m in self.children():
            m.train(mode)
        for m in self.ts_nets[:-1]:
            m.eval()
        return self

    def eval(self):
        return self.train(False)


    def freeze_old_backbones(self):
        for p in self.ts_nets[:-1].parameters():
            p.requires_grad_(False)
        if hasattr(self, 'ts_adapters'):
            for p in self.ts_adapters[:-1].parameters():
                p.requires_grad_(False)
        for net in self.ts_nets[:-1]:
            net.eval()