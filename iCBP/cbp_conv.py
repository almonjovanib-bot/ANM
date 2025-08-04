import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from typing import Union
from torch.nn.init import calculate_gain
from iCBP.cbp_linear import call_reinit, log_features, get_layer_bound


class CBPConv(nn.Module):
    def __init__(
            self,
            in_layer: nn.Conv2d,
            out_layer: [nn.Conv2d, nn.Linear],
            ln_layer: nn.LayerNorm = None,
            bn_layer: nn.BatchNorm2d = None,
            num_last_filter_outputs=1,
            replacement_rate=1e-5,
            maturity_threshold=1000,
            init='kaiming',
            act_type='relu',
            util_type='contribution',
            decay_rate=0,
            class_incremental=True,       
            incremental_factor=0.5         
    ):
        super().__init__()
        if type(in_layer) is not nn.Conv2d:
            raise Warning("Make sure in_layer is a convolutional layer")
        if type(out_layer) not in [nn.Linear, nn.Conv2d]:
            raise Warning("Make sure out_layer is a convolutional or linear layer")

        """
        Define algorithm hyperparameters
        """
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.features = None
        self.num_last_filter_outputs = num_last_filter_outputs

        self.class_incremental = class_incremental
        if self.class_incremental:
            self.effective_replacement_rate = self.replacement_rate * incremental_factor
            self.effective_maturity_threshold = max(1, int(self.maturity_threshold / incremental_factor))
        else:
            self.effective_replacement_rate = self.replacement_rate
            self.effective_maturity_threshold = self.maturity_threshold

        """
        Register hooks
        """
        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        self.in_layer = in_layer
        self.out_layer = out_layer
        self.ln_layer = ln_layer
        self.bn_layer = bn_layer
        """
        Initialize utility and age for all features/neurons
        """
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_channels), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_channels), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
        """
        Calculate uniform distribution bounds for random initialization of weights
        """
        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=calculate_gain(nonlinearity=act_type))

    def forward(self, _input):
        return _input

    def get_features_to_reinit(self):
        """
        Returns: indices of features to be replaced
        """
        features_to_replace_input_indices = torch.empty(0, dtype=torch.long, device=self.util.device)
        features_to_replace_output_indices = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1
        """
        Calculate number of features meeting adjusted maturity condition
        """
        eligible_feature_indices = torch.where(self.ages > self.effective_maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:  
            return features_to_replace_input_indices, features_to_replace_output_indices

        num_new_features_to_replace = self.effective_replacement_rate * eligible_feature_indices.shape[0]
        self.accumulated_num_features_to_replace += num_new_features_to_replace
        if self.accumulated_num_features_to_replace < 1:    
            return features_to_replace_input_indices, features_to_replace_output_indices

        num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
        self.accumulated_num_features_to_replace -= num_new_features_to_replace
        """
        Compute feature utility
        """
        if isinstance(self.out_layer, torch.nn.Linear):
            output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0).view(-1, self.num_last_filter_outputs)
            self.util.data = (output_weight_mag * self.features.abs().mean(dim=0).view(-1, self.num_last_filter_outputs)).mean(dim=1)
        elif isinstance(self.out_layer, torch.nn.Conv2d):
            output_weight_mag = self.out_layer.weight.data.abs().mean(dim=(0, 2, 3))
            self.util.data = output_weight_mag * self.features.abs().mean(dim=(0, 2, 3))
        """
        Select features with smallest utility
        """
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace_input_indices, features_to_replace_output_indices = new_features_to_replace, new_features_to_replace

        if isinstance(self.in_layer, torch.nn.Conv2d) and isinstance(self.out_layer, torch.nn.Linear):
            features_to_replace_output_indices = (
                    (new_features_to_replace * self.num_last_filter_outputs).repeat_interleave(self.num_last_filter_outputs) +
                    torch.tensor([i for i in range(self.num_last_filter_outputs)], device=self.util.device).repeat(new_features_to_replace.size()[0])
            )
        return features_to_replace_input_indices, features_to_replace_output_indices

    def reinit_features(self, features_to_replace_input_indices, features_to_replace_output_indices):
        """
        Reset input and output weights corresponding to low-utility features
        """
        with torch.no_grad():
            num_features_to_replace = features_to_replace_input_indices.shape[0]
            if num_features_to_replace == 0: 
                return
            self.in_layer.weight.data[features_to_replace_input_indices, :] = torch.empty_like(
                self.in_layer.weight.data[features_to_replace_input_indices, :]
            ).uniform_(-self.bound, self.bound)
            if self.in_layer.bias is not None:
                self.in_layer.bias.data[features_to_replace_input_indices].zero_()

            self.out_layer.weight.data[:, features_to_replace_output_indices].zero_()
            self.ages[features_to_replace_input_indices].zero_()

            """
            Reset corresponding batch normalization / layer normalization parameters
            """
            if self.bn_layer is not None:
                self.bn_layer.bias.data[features_to_replace_input_indices].zero_()
                self.bn_layer.weight.data[features_to_replace_input_indices].fill_(1.0)
                self.bn_layer.running_mean.data[features_to_replace_input_indices].zero_()
                self.bn_layer.running_var.data[features_to_replace_input_indices].fill_(1.0)
            if self.ln_layer is not None:
                self.ln_layer.bias.data[features_to_replace_input_indices].zero_()
                self.ln_layer.weight.data[features_to_replace_input_indices].fill_(1.0)

    def reinit(self):
        """
        Perform selective reset operation
        """
        features_to_replace_input_indices, features_to_replace_output_indices = self.get_features_to_reinit()
        self.reinit_features(features_to_replace_input_indices, features_to_replace_output_indices)
