from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

#from torchinfo import summary

from typing import NamedTuple, List, Callable
from collections import OrderedDict

from avalanche.training import utils as avl_utils


def disable_bn_tracking(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = False
    return model


def freeze_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    return model


def set_batchnorm_layers_to_eval(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = True
    return model

def set_batchnorm_track_statistics(model, do_track_statistics):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = do_track_statistics
    return model


def unfreeze_batchnorm_layers(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            for param in module.parameters():
                param.requires_grad = True
    return model

def freeze_norm_layers(model):
    """
    Freeze all normalization layers in the model.
    Including BatchNorm and LayerNorm
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    return model

def unfreeze_norm_layers(model):
    """
    Unfreeze all normalization layers in the model.
    Including BatchNorm and LayerNorm
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            print("Unfreezing layer:", module)
            module.train()
            for param in module.parameters():
                param.requires_grad = True
            #print("weights: ", module.weight.shape, module.weight.requires_grad)
            #print("bias:", module.bias.shape, module.bias.requires_grad)
    return model


def replace_BN(model, norm_type, max_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            # Replace BatchNorm with GroupNorm
            if norm_type == "group_norm":
                # Choose the largest divisor of num_features <= max_groups
                num_groups = min(max_groups, num_features)
                while num_features % num_groups != 0:
                    num_groups -= 1
                new_module = nn.GroupNorm(num_groups, num_features)
                # Copy over weight and bias
                # new_module.weight = module.weight
                # new_module.bias = module.bias
            elif norm_type == "layer_norm":
                new_module = nn.LayerNorm(num_features)
                print("LayerNorm: ", new_module)
                import sys; sys.exit()
            elif norm_type == "instance_norm":
                new_module = nn.InstanceNorm2d(num_features)
            else:
                raise ValueError(f"Invalid norm type {norm_type}. Choose from 'group_norm', 'layer_norm', 'instance_norm'")
            setattr(model, name, new_module)
        elif isinstance(module, nn.BatchNorm1d):
            num_features = module.num_features    
            # Replace BatchNorm with GroupNorm
            if norm_type == "group_norm":
                # Choose the largest divisor of num_features <= max_groups
                num_groups = min(max_groups, num_features)
                while num_features % num_groups != 0:
                    num_groups -= 1
                new_module = nn.GroupNorm(num_groups, num_features)
                # Copy over weight and bias
                # new_module.weight = module.weight
                # new_module.bias = module.bias
            elif norm_type == "layer_norm":
                new_module = nn.LayerNorm(num_features)
            elif norm_type == "instance_norm":
                new_module = nn.InstanceNorm1d(num_features)
            else:
                raise ValueError(f"Invalid norm type {norm_type}. Choose from 'group_norm', 'layer_norm', 'instance_norm'")
            setattr(model, name, new_module)
        else:
            # Recursively apply to child modules
            replace_BN(module, norm_type, max_groups)
    return

def replace_BN_by_GN(model, max_groups=32):
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            num_features = module.num_features
            # Choose the largest divisor of num_features <= max_groups
            num_groups = min(max_groups, num_features)
            while num_features % num_groups != 0:
                num_groups -= 1
            
            # Replace BatchNorm with GroupNorm
            new_module = nn.GroupNorm(num_groups, num_features)
            # Copy over weight and bias
            new_module.weight = module.weight
            new_module.bias = module.bias
            setattr(model, name, new_module)
        else:
            # Recursively apply to child modules
            replace_BN_by_GN(module, max_groups)
    return


def get_feat_size(block, spatial_size, in_channels=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.
    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    Source: https://github.com/TimmHess/OCDVAEContinualLearning/blob/master/lib/Models/architectures.py
    """

    block_device = next(block.parameters()).device
    x = torch.randn(2, in_channels, spatial_size, spatial_size).to(block_device)
    out = block(x)
    if len(out.size()) == 2: # NOTE: catches the case where the block is a linear layer
        num_feat = out.size(1)
        spatial_dim_x = 1
        spatial_dim_y = 1
    else:
        num_feat = out.size(1)
        spatial_dim_x = out.size(2)
        spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y

def initialize_weights(m) -> None:
    """
    Initilaize weights of model m.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    return 

def reinit_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()

def reinit_after(model, reinit_after=None, module_prefix=""):
    do_skip = True # NOTE: flag to skip the first layers in reinitialization
    for param_def in get_layers_and_params(model, prefix=module_prefix):
        if reinit_after in param_def.layer_name: # NOTE: if reinit_after is None, nothing is reinitialized!
            do_skip = False
        
        if do_skip: # NOTE: this will skip the first n layers in execution
            print("Skipping layer {}".format(param_def.layer_name))
            continue
        # TODO: re-add layer filter option (it was too annoying to implement)
        reinit_weights(param_def.layer)
        print("Reinitialized {}".format(param_def.layer_name))
    return
    

class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor

def get_layers_and_params(model, prefix=''):
    """
    Adapted from AvalancheCL lib
    """
    result: List[LayerAndParameter] = []
    layer_name: str
    layer: Module
    for layer_name, layer in model.named_modules():
        if layer == model:
            continue
        if isinstance(layer, nn.Sequential): # NOTE: cannot include Sequentials because this is basically a repetition of parameter listings
            continue
        layer_complete_name = prefix + layer_name + "."
        layers_and_params = avl_utils.get_layers_and_params(layer, prefix=layer_complete_name) #NOTE: this calls to avalanche function! (not self)
        result += layers_and_params

    unique_layers = OrderedDict()
    for param_def in result:
        if param_def.layer_name not in unique_layers:
            unique_layers[param_def.layer_name] = param_def
    unique_layers = list(unique_layers.values())
    return unique_layers


def freeze(model, bn_eval=True):
    for module in model.modules():
        if bn_eval:
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
                module.eval()
        for param in module.parameters():
            param.requires_grad = False
    return

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return

def freeze_from_to(
        model: Module,
        freeze_from_layer: str = None,
        freeze_until_layer: str = None,
        set_eval_mode: bool = True,
        set_requires_grad_false: bool = True,
        module_prefix: str = ""):
    
    frozen_layers = set()
    frozen_parameters = set()
    
    layer_and_params = get_layers_and_params(model, prefix=module_prefix) 

    is_freezing = False # NOTE: status flag to determine if we are freezing or not
    
    for param_def in layer_and_params:
        print("freeze_from_to:: param_def: ", param_def.layer_name)
        #print("module:", param_def.layer, isinstance(param_def.layer, torch.nn.modules.batchnorm._BatchNorm))
        # Check if first layer to freeze is reached
        if not is_freezing and ((freeze_from_layer is None) or (freeze_from_layer in param_def.layer_name)):
            is_freezing = True
            print("Start freezing, including:", param_def.layer_name)

        # Check if last layer to freeze was reached
        if is_freezing and (freeze_until_layer is not None) and (freeze_until_layer in param_def.layer_name): 
            print("Stop freezing layers, not freezing:", param_def.layer_name)
            is_freezing = False
        
        if is_freezing:
            if set_requires_grad_false:
                param_def.parameter.requires_grad = False
                frozen_parameters.add(param_def.parameter_name)
            if set_eval_mode:
                param_def.layer.eval()
                frozen_layers.add(param_def.layer_name)
                
    return frozen_layers, frozen_parameters

def freeze_up_to(
            model: Module,
            freeze_until_layer: str = None,
            set_eval_mode: bool = True,
            set_requires_grad_false: bool = True,
            layer_filter: Callable[[LayerAndParameter], bool] = None,
            module_prefix: str = ""):
    """
    A simple utility that can be used to freeze a model.
    :param model: The model.
    :param freeze_until_layer: If not None, the freezing algorithm will continue
        (proceeding from the input towards the output) until the specified layer
        is encountered. The given layer is excluded from the freezing procedure.
    :param set_eval_mode: If True, the frozen layers will be set in eval mode.
        Defaults to True.
    :param set_requires_grad_false: If True, the autograd engine will be
        disabled for frozen parameters. Defaults to True.
    :param layer_filter: A function that, given a :class:`LayerParameter`,
        returns `True` if the parameter must be frozen. If all parameters of
        a layer are frozen, then the layer will be set in eval mode (according
        to the `set_eval_mode` parameter. Defaults to None, which means that all
        parameters will be frozen.
    :param module_prefix: The model prefix. Do not use if non strictly
        necessary.
    :return:
    """
    print("entering freeze-up-to function...")
    frozen_layers = set()
    frozen_parameters = set()

    to_freeze_layers = dict()
    layer_and_params = get_layers_and_params(model, prefix=module_prefix) 
    for param_def in layer_and_params:
        #print(param_def)
        if(freeze_until_layer is not None
            and freeze_until_layer in param_def.layer_name): # freeze_until_layer == param_def.layer_name
            print("Will not freeze:", param_def.layer_name)
            print("Will stop freezing layers...")
            break

        print("Will freeze:", param_def.layer_name)
        freeze_param = layer_filter is None or layer_filter(param_def)
        if freeze_param:
            if set_requires_grad_false:
                param_def.parameter.requires_grad = False
                frozen_parameters.add(param_def.parameter_name)

            if param_def.layer_name not in to_freeze_layers:
                to_freeze_layers[param_def.layer_name] = (True, param_def.layer)
        else:
            # Don't freeze this parameter -> do not set eval on the layer
            to_freeze_layers[param_def.layer_name] = (False, None)

    if set_eval_mode:
        for layer_name, layer_result in to_freeze_layers.items():
            if layer_result[0]:
                layer_result[1].eval()
                frozen_layers.add(layer_name)
                
    return frozen_layers, frozen_parameters


def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model.

    This has the same effect as permanently executing the model within a `torch.no_grad()`
    context. Use this method to disable gradient computation and therefore
    training for a model.
    """
    for param in model.parameters():
        param.requires_grad = False


# def get_model_summary(model, input_size, show_backbone_param_names=False, device='cpu'):
#     summary(model.feature_extractor, input_size=input_size, device=device) 
    
#     if show_backbone_param_names:
#         print("Modules:")
#         for module in model.feature_extractor.named_modules():
#             print(module)
#         print("\nStopping execution here! Remove the 'show_backbone_param_names' flag to continue!")
#         import sys;sys.exit()


def get_param_groups(model, lr, weight_decay):
    """
    optimizer = torch.optim.AdamW(get_param_groups(model))
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check for bias, bn, or layernorm
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {'params': decay, 'weight_decay': weight_decay, 'lr': lr},
        {'params': no_decay, 'weight_decay': 0.0, 'lr': lr}
    ]


