import copy

import torch
from timm.models.layers import trunc_normal_
from torch import nn
from modules import NaiveClassIncrementalNetwork
from modules import Accuracy, MeanMetric, CatMetric, select_metrics, forward_metrics, get_metrics
from modules import optimizer_dispatch, scheduler_dispatch, get_loaders
from utils.configuration import load_configs
from utils.funcs import parameter_count

EPSILON = 1e-8

class DyTox(nn.Module):
    """"DyTox for the win!

    :param transformer: The base transformer.
    :param nb_classes: Thhe initial number of classes.
    :param individual_classifier: Classifier config, DyTox is in `1-1`.
    :param head_div: Whether to use the divergence head for improved diversity.
    :param head_div_mode: Use the divergence head in TRaining, FineTuning, or both.
    :param joint_tokens: Use a single TAB forward with masked attention (faster but a bit worse).
    """
    def __init__(
        self,
        transformer,
        nb_classes,
        individual_classifier='',
        head_div=False,
        head_div_mode=['tr', 'ft'],
        joint_tokens=False,
        resnet=False
    ):
        super().__init__()

        self.nb_classes = nb_classes
        self.embed_dim = transformer.embed_dim
        self.individual_classifier = individual_classifier
        self.use_head_div = head_div
        self.head_div_mode = head_div_mode
        self.head_div = None
        self.joint_tokens = joint_tokens
        self.in_finetuning = False

        self.use_resnet = resnet

        self.nb_classes_per_task = [nb_classes]

        if self.use_resnet:
            print('ResNet18 backbone for ens')
            self.backbone = resnet18()
            self.backbone.head = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 504, kernel_size=1),
                nn.BatchNorm2d(504),
                nn.ReLU(inplace=True)
            )
            self.backbone.avgpool = nn.Identity()
            self.backbone.layer4 = nn.Identity()
            #self.backbone.layer4 = self.backbone._make_layer_nodown(
            #    256, 512, 2, stride=1, dilation=2
            #)
            self.backbone = self.backbone.cuda()
            self.backbone.embed_dim = 504
            self.embed_dim = self.backbone.embed_dim

            self.tabs = nn.ModuleList([
                Block(
                    dim=self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                    attention_type=ClassAttention
                ).cuda()
            ])
            self.tabs[0].reset_parameters()

            token = nn.Parameter(torch.zeros(1, 1, self.embed_dim).cuda())
            trunc_normal_(token, std=.02)
            self.task_tokens = nn.ParameterList([token])
        else:
            self.patch_embed = transformer.patch_embed
            self.pos_embed = transformer.pos_embed
            self.pos_drop = transformer.pos_drop
            self.sabs = transformer.blocks[:transformer.local_up_to_layer]
            self.tabs = transformer.blocks[transformer.local_up_to_layer:]
            self.task_tokens = nn.ParameterList([transformer.cls_token])

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()

    def end_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def add_model(self, nb_new_classes):
        """Expand model as per the DyTox framework given `nb_new_classes`.

        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        # Class tokens ---------------------------------------------------------
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        # ----------------------------------------------------------------------

        # Diversity head -------------------------------------------------------
        if self.use_head_div:
            self.head_div = ContinualClassifier(
                self.embed_dim, self.nb_classes_per_task[-1] + 1
            ).cuda()
        # ----------------------------------------------------------------------

        # Classifier -----------------------------------------------------------
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head.append(
                ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()
        # ----------------------------------------------------------------------

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.

        By default, DyTox is in 1-1.
        """
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.task_tokens, requires_grad=requires_grad)
            elif name == 'sab':
                if self.use_resnet:
                    self.backbone.eval()
                    cutils.freeze_parameters(self.backbone, requires_grad=requires_grad)
                else:
                    self.sabs.eval()
                    cutils.freeze_parameters(self.patch_embed, requires_grad=requires_grad)
                    cutils.freeze_parameters(self.pos_embed, requires_grad=requires_grad)
                    cutils.freeze_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.freeze_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'ca': self.tabs.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                              if self.individual_classifier else \
                              self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
            'head_div': self.head_div.parameters() if self.head_div is not None else None
        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.task_tokens)):
                for j in range(i + 1, len(self.task_tokens)):
                    dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        return log

    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses

    def forward_features(self, x):
        # Shared part, this is the ENCODER
        B = x.shape[0]

        if self.use_resnet:
            x, self.feats = self.backbone.forward_tokens(x)
        else:
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            self.feats = []
            for blk in self.sabs:
                x, attn, v = blk(x)
                self.feats.append(x)
            self.feats.pop(-1)

        # Specific part, this is what we called the "task specific DECODER"
        if self.joint_tokens:
            return self.forward_features_jointtokens(x)

        tokens = []
        attentions = []
        mask_heads = None

        for task_token in self.task_tokens:
            task_token = task_token.expand(B, -1, -1)

            for blk in self.tabs:
                task_token, attn, v = blk(torch.cat((task_token, x), dim=1), mask_heads=mask_heads)

            attentions.append(attn)
            tokens.append(task_token[:, 0])

        self._class_tokens = tokens
        return tokens, tokens[-1], attentions

    def forward_features_jointtokens(self, x):
        """Method to do a single TAB forward with all task tokens.

        A masking is used to avoid interaction between tasks. In theory it should
        give the same results as multiple TAB forward, but in practice it's a little
        bit worse, not sure why. So if you have an idea, please tell me!
        """
        B = len(x)

        task_tokens = torch.cat(
            [task_token.expand(B, 1, -1) for task_token in self.task_tokens],
            dim=1
        )

        for blk in self.tabs:
            task_tokens, _, _ = blk(
                torch.cat((task_tokens, x), dim=1),
                task_index=len(self.task_tokens),
                attn_mask=True
            )

        if self.individual_classifier in ('1-1', '1-n'):
            return task_tokens.permute(1, 0, 2), task_tokens[:, -1], None
        return task_tokens.view(B, -1), task_tokens[:, -1], None

    def forward_classifier(self, tokens, last_token):
        """Once all task embeddings e_1, ..., e_t are extracted, classify.

        Classifier has different mode based on a pattern x-y:
        - x means the number of task embeddings in input
        - y means the number of task to predict

        So:
        - n-n: predicts all task given all embeddings
        But:
        - 1-1: predict 1 task given 1 embedding, which is the 'independent classifier' used in the paper.

        :param tokens: A list of all task tokens embeddings.
        :param last_token: The ultimate task token embedding from the latest task.
        """
        logits_div = None

        if self.individual_classifier != '':
            logits = []

            for i, head in enumerate(self.head):
                if self.individual_classifier in ('1-n', '1-1'):
                    logits.append(head(tokens[i]))
                else:  # n-1, n-n
                    logits.append(head(torch.cat(tokens[:i+1], dim=1)))

            if self.individual_classifier in ('1-1', 'n-1'):
                logits = torch.cat(logits, dim=1)
            else:  # 1-n, n-n
                final_logits = torch.zeros_like(logits[-1])
                for i in range(len(logits)):
                    final_logits[:, :logits[i].shape[1]] += logits[i]

                for i, c in enumerate(self.nb_classes_per_task):
                    final_logits[:, :c] /= len(self.nb_classes_per_task) - i

                logits = final_logits
        elif isinstance(tokens, torch.Tensor):
            logits = self.head(tokens)
        else:
            logits = self.head(torch.cat(tokens, dim=1))

        if self.head_div is not None and eval_training_finetuning(self.head_div_mode, self.in_finetuning):
            logits_div = self.head_div(last_token)  # only last token

        return {
            'logits': logits,
            'div': logits_div,
            'tokens': tokens
        }

    def forward(self, x):
        tokens, last_token, _ = self.forward_features(x)
        return self.forward_classifier(tokens, last_token)


class DyToxIncremental(HerdingIndicesLearner):
    def __init__(self, data_manager, configs: dict, device, distributed=None) -> None:
        super().__init__(data_manager, configs, device, distributed)

        # 使用 DyTox 初始化模型
        self._init_network(self.configs.get('transformer_configs', dict()), self.configs.get('network_configs', dict()))

        if self.distributed is not None:
            self._init_ddp()

        self._init_loggers()
        self.print_logger.info(configs)

        self.print_logger.info(f'class order: {self.data_manager.class_order.tolist()}')
        self.ordered_index_map = torch.from_numpy(self.data_manager.ordered_index_map).to(self.device)

        self.teacher_network = None

    def _init_network(self, transformer_configs, network_configs):
        # 用 DyTox 替代 iCaRL 的 NaiveClassIncrementalNetwork
        self.network = DyTox(
            transformer_configs['transformer'], 
            self.data_manager.task_num_cls[0], 
            individual_classifier=self.configs.get('individual_classifier', ''),
            head_div=self.configs.get('head_div', False),
            head_div_mode=self.configs.get('head_div_mode', ['tr', 'ft']),
            joint_tokens=self.configs.get('joint_tokens', False),
            resnet=self.configs.get('use_resnet', False)
        ).to(self.device)

        self.local_network = self.network # for ddp compatibility

    def _init_ddp(self):
        self.configs['trainloader_params']['batch_size'] //= self.distributed['world_size']
        torch.distributed.barrier()

    def _model_to_ddp(self):
        self.network = nn.parallel.DistributedDataParallel(
            self.local_network, device_ids=[self.distributed['rank']], find_unused_parameters=self.configs['debug']
        )

    def train(self) -> None:
        # 其余训练过程不变，依然使用 DyTox 网络进行训练
        self.update_state(run_state='train', num_tasks=self.data_manager.num_tasks)
        self.run_metrics = {
            'acc1_curve': CatMetric(sync_on_compute=False).to(self.device),
            'acc5_curve': CatMetric(sync_on_compute=False).to(self.device),
            'nme1_curve': CatMetric(sync_on_compute=False).to(self.device),
            'nme5_curve': CatMetric(sync_on_compute=False).to(self.device),
            'avg_acc1': MeanMetric().to(self.device),
            'avg_acc5': MeanMetric().to(self.device),
            'avg_nme1': MeanMetric().to(self.device),
            'avg_nme5': MeanMetric().to(self.device)
        }

        for task_id, (task_train, task_test) in enumerate(self.data_manager.tasks):
            cur_task_num_classes = self.data_manager.task_num_cls[task_id]
            sofar_num_classes = sum(self.data_manager.task_num_cls[:task_id+1])
            self.update_state(cur_task=task_id+1, cur_task_num_classes=cur_task_num_classes, sofar_num_classes=sofar_num_classes)
            
            if task_id > 0:
                self.teacher_network = self.local_network.freezed_copy()

            self.local_network.update_network(cur_task_num_classes)
            if self.configs['ckpt_path'] is not None and self.configs['ckpt_task'] is not None and task_id + 1 <= self.configs['ckpt_task']:
                if task_id + 1 == self.configs['ckpt_task']:
                    self._load_checkpoint(self.configs['ckpt_path'])
                continue
            
            total, trainable = parameter_count(self.local_network)
            self.print_logger.info(f'{self._get_status()} | parameters: {total} in total, {trainable} trainable.')
            
            if self.distributed is not None:
                self._model_to_ddp()
            
            # add memory into training set.
            memory_indices = self.get_memory()
            new_indices = np.concatenate((task_train.indices, memory_indices))
            task_train.indices = new_indices

            # get dataloaders
            if self.configs['ffcv']:
                from modules.data.ffcv.loader import get_ffcv_loaders
                train_loader, test_loader = get_ffcv_loaders(task_train, task_test, self.configs['trainloader_params'], self.configs['testloader_params'], self.device, self.configs, self.distributed is not None)
            else:
                train_loader, test_loader = get_loaders(task_train, task_test, self.configs['trainloader_params'], self.configs['testloader_params'], self.distributed)

            self.train_task(train_loader, test_loader)
            self.print_logger.info(f"Adjust memory to {self.num_exemplars_per_class} per class ({self.num_exemplars_per_class * self.state['sofar_num_classes']} in total).")
            self.reduce_memory()
            self.update_memory()
            
            # evaluation as task end
            results = self.eval_epoch(test_loader)
            forward_metrics(select_metrics(self.run_metrics, 'acc1'), results['eval_acc1'])
            forward_metrics(select_metrics(self.run_metrics, 'acc5'), results['eval_acc5'])
            forward_metrics(select_metrics(self.run_metrics, 'nme1'), results['eval_nme1'])
            forward_metrics(select_metrics(self.run_metrics, 'nme5'), results['eval_nme5'])
            self.print_logger.success(f'{self._get_status()}\n├> {self._metric_repr(results | get_metrics(self.run_metrics))}')
            if self.configs['ckpt_dir'] is not None and not self.configs['disable_save_ckpt'] and task_id + 1 in self.configs['save_ckpt_tasks']:
                self._save_checkpoint()

        results = get_metrics(self.run_metrics)
        self.print_logger.success(f'{self._get_status()}\n├> {self._metric_repr(results)}')
        self.update_state(run_state='finished')

    # 其他方法保持不变...


    def train_task(self, train_loader, test_loader):
        # get optimizer and lr scheduler
        if self.state['cur_task'] == 1:
            optimizer = optimizer_dispatch(self.local_network.parameters(), self.configs['init_optimizer_configs'])
            scheduler = scheduler_dispatch(optimizer, self.configs['init_scheduler_configs'])

            num_epochs = self.configs['init_epochs'] # FIXME: config category
        else:
            optimizer = optimizer_dispatch(self.local_network.parameters(), self.configs['inc_optimizer_configs'])
            scheduler = scheduler_dispatch(optimizer, self.configs['inc_scheduler_configs'])

            num_epochs = self.configs['inc_epochs'] # FIXME: config category

        # >>> @after_train_task_setups
        if self.configs['debug']:
            num_epochs = 5
        self.update_state(cur_task_num_epochs=num_epochs)
        # <<< @after_train_task_setups

        rank = 0 if self.distributed is None else self.distributed['rank']
        prog_bar = tqdm(range(num_epochs), desc=f"Task {self.state['cur_task']}/{self.state['num_tasks']}") if rank == 0 else range(num_epochs)
        for epoch in prog_bar:
            # >>> @train_epoch_start
            self.update_state(cur_epoch=epoch + 1, num_batches=len(train_loader))
            self.add_state(accumulated_cur_epoch=1)
            if self.distributed is not None and not self.configs['ffcv']:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)
            # <<< @train_epoch_start

            train_results = self.train_epoch(train_loader, optimizer, scheduler)
            
            # >>> @train_epoch_end
            if epoch % self.configs['eval_interval'] == 0:
                eval_results = self.eval_epoch(test_loader)
                self.print_logger.info(f'{self._get_status()} | {self._metric_repr(train_results)} {self._metric_repr(eval_results)}')
            else:
                self.print_logger.info(f'{self._get_status()} | {self._metric_repr(train_results)}')
            # <<< @train_epoch_end
        if rank == 0:
            prog_bar.close()

    def train_epoch(self, train_loader, optimizer, scheduler):
        self.network.train()
        num_classes = self.state['sofar_num_classes'].item()
        if self.teacher_network is None:
            metrics = {
                'loss': MeanMetric().to(self.device),
                'train_acc1': Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
            }
        else:
            metrics = {
                'cls_loss': MeanMetric().to(self.device),
                'kd_loss': MeanMetric().to(self.device),
                'train_acc1': Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
            }
        for batch, batch_data in enumerate(train_loader):
            batch_data = tuple(data.to(self.device, non_blocking=True) for data in batch_data)

            samples, targets = batch_data
            targets = self.ordered_index_map[targets.flatten()] # map to continual class id.
            # self.print_logger.debug(f'train {batch}/{len(train_loader)}', samples.device, targets.device)
            # self.print_logger.debug(f'batch shape {samples.shape}')

            # >>> @train_batch_start
            self.update_state(cur_batch=batch+1)
            # <<< @train_batch_start

            # >>> @train_forward
            logits = self.network(samples.contiguous())["logits"]

            if self.teacher_network is None:
                loss = F.cross_entropy(logits, targets)
            else:
                cls_loss = F.cross_entropy(logits, targets)

                old_logits = self.teacher_network(samples.contiguous())["logits"]
                distill_dims = old_logits.shape[-1]
                T = self.configs['kd_temp']
                kd_loss = F.kl_div(
                    (logits[:, :distill_dims] / T).log_softmax(dim=-1),
                    (old_logits / T).softmax(dim=-1).detach(),
                    reduction='batchmean'
                )
                loss = cls_loss + kd_loss
            # <<< @train_forward

            # >>> @train_backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # <<< @train_backward

            # >>> @train_batch_end
            if self.teacher_network is None:
                metrics['loss'].update(loss.detach())
            else:
                metrics['cls_loss'].update(cls_loss.detach())
                metrics['kd_loss'].update(kd_loss.detach())
            metrics['train_acc1'].update(logits.detach(), targets.detach())
            # <<< @train_batch_end
        
        scheduler.step()
        train_results = get_metrics(metrics)
        return train_results

    @torch.no_grad()
    def eval_epoch(self, data_loader):
        # >>> @eval_start
        prev_run_state = self.state.get('run_state')
        self.update_state(run_state='eval')
        self.network.eval()
        # <<< @eval_start

        num_classes = self.state['sofar_num_classes'].item()
        acc_metrics = {
            'eval_acc1': Accuracy(task='multiclass', num_classes=num_classes).to(self.device),
            'eval_acc5': Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(self.device),
            'eval_acc1_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes).to(self.device),
            'eval_acc5_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes, top_k=5).to(self.device),
        }
        if len(self.class_means) == self.state['sofar_num_classes']:
            nme_metrics = {
                'eval_nme1': Accuracy(task='multiclass', num_classes=num_classes).to(self.device),
                'eval_nme5': Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(self.device),
                'eval_nme1_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes).to(self.device),
                'eval_nme5_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes, top_k=5).to(self.device),
            }

        # >>> @eval_epoch_start
        self.update_state(eval_num_batches=len(data_loader))
        # <<< @eval_epoch_start

        for batch, batch_data in enumerate(data_loader):
            batch_data = tuple(data.to(self.device, non_blocking=True) for data in batch_data)

            # >>> @eval_batch_start
            self.update_state(eval_cur_batch=batch+1)
            # <<< @eval_batch_start

            samples, targets = batch_data
            targets = self.ordered_index_map[targets.flatten()] # map to continual class id.
            # self.print_logger.debug(f'eval {batch}/{len(data_loader)}', samples.device, targets.device)
            outs = self.network(samples.contiguous())
            logits = outs['logits']
            forward_metrics(acc_metrics, logits, targets)

            if len(self.class_means) == self.state['sofar_num_classes']:
                features = outs['features']
                features /= features.norm(dim=-1, keepdim=True) + EPSILON
                dists = torch.cdist(features, torch.stack(self.class_means))
                forward_metrics(nme_metrics, -dists, targets)

        acc_metric_results = get_metrics(acc_metrics)
        if len(self.class_means) == self.state['sofar_num_classes']:
            nme_metric_results = get_metrics(nme_metrics)
        else:
            nme_metric_results = dict()

        # >>> @eval_end
        self.update_state(run_state=prev_run_state)
        # >>> @eval_end

        return acc_metric_results | nme_metric_results
    
    def evaluate(self):
        # >>> @evaluate_start
        self.update_state(run_state='evaluate', num_tasks=self.data_manager.num_tasks)
        self.run_metrics = {
            'acc1_curve': CatMetric(sync_on_compute=False).to(self.device),
            'acc5_curve': CatMetric(sync_on_compute=False).to(self.device),
            'nme1_curve': CatMetric(sync_on_compute=False).to(self.device),
            'nme5_curve': CatMetric(sync_on_compute=False).to(self.device),
            'avg_acc1': MeanMetric().to(self.device),
            'avg_acc5': MeanMetric().to(self.device),
            'avg_nme1': MeanMetric().to(self.device),
            'avg_nme5': MeanMetric().to(self.device)
        }
        # <<< @evaluate_start

        for task_id, (task_train, task_test) in enumerate(self.data_manager.tasks):
            # >>> @evaluate_task_start
            cur_task_num_classes = self.data_manager.task_num_cls[task_id]
            sofar_num_classes = sum(self.data_manager.task_num_cls[:task_id+1])
            self.update_state(cur_task=task_id+1, cur_task_num_classes=cur_task_num_classes, sofar_num_classes=sofar_num_classes)

            self.local_network.update_network(cur_task_num_classes)
            ckpt_path = self.configs['ckpt_paths'][task_id]
            self._load_checkpoint(ckpt_path)
            
            total, trainable = parameter_count(self.local_network)
            self.print_logger.info(f'{self._get_status()} | parameters: {total} in total')
            
            if self.distributed is not None:
                self._model_to_ddp()

            # get dataloaders
            if self.configs['ffcv']:
                from modules.data.ffcv.loader import get_ffcv_loaders
                train_loader, test_loader = get_ffcv_loaders(task_train, task_test, self.configs['trainloader_params'], self.configs['testloader_params'], self.device, self.configs, self.distributed is not None)
            else:
                train_loader, test_loader = get_loaders(task_train, task_test, self.configs['trainloader_params'], self.configs['testloader_params'], self.distributed)
            # <<< @evaluate_task_start

            results = self.eval_epoch(test_loader)

            # >>> @evaluate_task_end
            forward_metrics(select_metrics(self.run_metrics, 'acc1'), results['eval_acc1'])
            forward_metrics(select_metrics(self.run_metrics, 'acc5'), results['eval_acc5'])
            forward_metrics(select_metrics(self.run_metrics, 'nme1'), results['eval_nme1'])
            forward_metrics(select_metrics(self.run_metrics, 'nme5'), results['eval_nme5'])
            self.print_logger.success(f'{self._get_status()}\n├> {self._metric_repr(results | get_metrics(self.run_metrics))}')
            # <<< @evaluate_task_end
        
        # >>> @evaluate_end
        results = get_metrics(self.run_metrics)
        self.print_logger.success(f'{self._get_status()}\n├> {self._metric_repr(results)}')
        self.update_state(run_state='finished')
        # <<< @evaluate_end

    def _metric_repr(self, metric_results: dict):
        def merge_to_task(acc_per_cls):
            display_value = []
            accumuated_num_cls = 0
            for num_cls in self.data_manager.task_num_cls:
                tot_num_cls = accumuated_num_cls + num_cls
                if tot_num_cls > len(acc_per_cls):
                    break
                task_acc = acc_per_cls[accumuated_num_cls:tot_num_cls].mean()
                display_value.append(task_acc.item())
                accumuated_num_cls = tot_num_cls
            return display_value

        scalars = []
        vectors = []
        for key, value in metric_results.items():
            if 'acc' in key or 'nme' in key:
                display_value = value * 100
            else:
                display_value = value
            
            if value.dim() > 0:
                if 'per_class' in key:
                    display_value = merge_to_task(display_value)
                    key = key.replace('per_class', 'per_task')
                else:
                    display_value = display_value.cpu().tolist()
                [f"{v:.2f}" for v in display_value]
                r = f'{key} [{" ".join([f"{v:.2f}" for v in display_value])}]'
                vectors.append(r)
            else:
                r = f'{key} {display_value.item():.2f}'
                scalars.append(r)
        
        componets = []
        if len(scalars) > 0:
            componets.append(' '.join(scalars))
        if len(vectors) > 0:
            componets.append('\n├> '.join(vectors))
        s = '\n├> '.join(componets)
        return '\n└>'.join(s.rsplit('\n├>', 1))

    def _init_loggers(self):
        self.loguru_logger = LoguruLogger(self.configs, self.configs['disable_log_file'], tqdm_out=True)
        self.print_logger = self.loguru_logger.logger # the actual logger
    
    def _get_status(self):
        if self.distributed is None:
            rank, world_size = 0, 1
        else:
            rank, world_size = self.distributed['rank'], self.distributed['world_size']
        run_state = self.state.get('run_state')
        num_tasks = self.state.get('num_tasks')
        cur_task = self.state.get('cur_task')
        cur_task_num_classes = self.state.get('cur_task_num_classes')
        sofar_num_classes = self.state.get('sofar_num_classes')
        cur_task_num_epochs = self.state.get('cur_task_num_epochs')
        cur_epoch = self.state.get('cur_epoch')
        num_batches = self.state.get('num_batches')
        cur_batch = self.state.get('cur_batch')
        eval_num_batches = self.state.get('eval_num_batches')
        eval_cur_batch = self.state.get('eval_cur_batch')

        if run_state == 'train':
            status = f"R{rank}T[{cur_task}/{num_tasks}]E[{cur_epoch}/{cur_task_num_epochs}] {run_state}"
        elif run_state == 'eval':
            status = f"R{rank}T[{cur_task}/{num_tasks}]E[{cur_epoch}/{cur_task_num_epochs}] {run_state}"
        
        return status

    def state_dict(self) -> dict:
        super_dict = super().state_dict()
        d = {
            "network_state_dict": self.local_network.state_dict(),
            "teacher_network_state_dict": self.teacher_network.state_dict() if self.teacher_network is not None else None,
            "run_metrics": {name: metric.state_dict() for name, metric in self.run_metrics.items()}
        }
        return super_dict | d
    
    def load_state_dict(self, d) -> None:
        super().load_state_dict(d)
        network_state_dict = d['network_state_dict']
        self.local_network.load_state_dict(network_state_dict)

        teacher_network_state_dict = d['teacher_network_state_dict']
        if teacher_network_state_dict is not None and self.teacher_network is not None:
            self.teacher_network.load_state_dict(teacher_network_state_dict)
        
        run_metrics = d.get('run_metrics', dict())
        for name, state_dict in run_metrics.items():
            self.run_metrics[name].load_state_dict(state_dict)

    def _save_checkpoint(self):
        save_dict = self.state_dict()

        cur_task = self.state['cur_task']
        num_tasks = self.state['num_tasks']
        dataset_name = self.data_manager.dataset_name
        task_name, scenario = self.data_manager.scenario.split(' ')
        method = self.configs['method']

        ckpt_file_name = f"{method}_{dataset_name}_{scenario}_[{cur_task}_{num_tasks}].ckpt"
        ckpt_dir = self.configs['ckpt_dir']
        ckpt_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        torch.save(save_dict, ckpt_dir / ckpt_file_name)
    
    def _load_checkpoint(self, path):
        ckpt = torch.load(path, self.device)
        self.load_state_dict(ckpt)