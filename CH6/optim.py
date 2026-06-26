import json
import torch

def get_num_layer_for_transformers(param_name, num_max_layer):
    """Part of layer-wise learning rate decay (LLRD), assigning a layer ID to a parameter name."""
    # print(f"In get_num_layer_for_transformers {param_name=}, {num_max_layer=}")
    if any(s in param_name for s in ('tok_emb', 'pos_emb')): return 0
    elif param_name.startswith('trf_blocks'):
        layer_id=int(param_name.split('.')[1]) # e.g., trf_blocks.11.att.out_proj.weight
        return layer_id+1
    else: return num_max_layer-1 # head gets the highest index

class LayerDecayValueAssigner(object):
    """Part of layer-wise learning rate decay (LLRD), to compute scale applied to learning rate per parameter/layer
    
    Return decay factor/scale that used to scale learning rate/weight decay so that the learning rate/weight decay decreases as we go 
    towards the input. In other words, the factor starts from 1. for the last layer and decreases as we move toward the input
        lr_layer=lr_base*factor where factor=`decay_rate`**(num_layer-layer-1)
    """
    def __init__(self, values): self.values=values
    def get_scale(self, layer_id): return self.values[layer_id]
    def get_layer_id(self, var_name): return get_num_layer_for_transformers(var_name, len(self.values))

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, verbose=False):
    """ Divide parameters into a set with weight_decay imposed and set without. The set that will not have weight decay includes those
    that are 1D parameters, bias, scale, and those in skip_list
    Args:
        model (nn.Module): Model
        weight_decay (float): Weight decay
        skip_list (sequence): Sequence of parameter names that will not be constrained through weight decay
    Returns:
        (list[dict]): List of parameter groups, each group is a dict having format {'weight_decay':float, 'params':[],
            'lr_scale':float}
    Reference: Reference: https://github.com/OpenGVLab/VideoMAEv2/blob/master/optim_factory.py#L56
    """
    parameter_group_names={} # storing parameter names
    parameter_group_vars={} # storing variable values/parameters
    
    for i, (name, param) in enumerate(model.named_parameters()):
        #print(i, name, '-'*20)
        if not param.requires_grad: continue # frozen weight
        if len(param.shape)==1 or name.endswith('.bias') or name.endswith('scale') or any(s in name for s in skip_list):
            #print(f"shape1, bias, scale, skip: {name=}, {param.shape=}")
            group_name="no_decay"
            this_weight_decay=0.
        else: 
            group_name='decay'
            this_weight_decay=weight_decay
        layer_id=None
        if get_num_layer is not None:
            layer_id=get_num_layer(name)
            group_name=f"layer_{layer_id}_{group_name}"
            #print(f"{layer_id=}, {group_name=}")
        if group_name not in parameter_group_names:
            scale=1.
            if get_layer_scale is not None: scale=get_layer_scale(layer_id)
            parameter_group_names[group_name]={"weight_decay":this_weight_decay, "params":[], "lr_scale":scale}
            parameter_group_vars[group_name]={"weight_decay":this_weight_decay, "params":[], "lr_scale":scale}
            #print(f"{scale=}")
            
        parameter_group_names[group_name]['params'].append(name)
        parameter_group_vars[group_name]['params'].append(param)
    
    if verbose: print(f"Param group {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())

def create_optimizer(model, weight_decay, lr, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, 
                     verbose=False):
    """Build and return AdamW optimizer, supporting layer-wise learning rate scaling and weight decay filtering
    
    It supports Layer-wise learning rate decay (LLRD) during fine tuning. The core idea is the earlier layers (closer to the input)
    should have a smaller learning rate than later layers (closer to the output) during fine tuning to preserve the foundational 
    features learned during pretraining, while allowing the head to adapt to the new task. LLRD is implemented based on the 
    `get_num_layer` and `get_layer_scale`. See `get_num_layer_for_transformers` and `LayerDecayValueAssigner`.
    Args:
        model (nn.Module): Model
        get_num_layer (callable): A function mapping a parameter name to its layer index. Use in LLRD
        get_layer_scale (callable): A function returning the specific learning rate scaling factor for a given layer index
        filter_bias_and_bn (bool): Whether to disable weight decay for all bias parameters and normalization layer weights
        skip_list (sequence, optional): A collection of parameter names that should be explicitly excluded from weight decay. 
            If None, the function attempts to call model.no_weight_decay() to retrieve the list
    Returns:
        (torch.optim.AdamW): An optimizer with parameters partitioned into specific groups for decay and scaling 
    """
    if weight_decay and filter_bias_and_bn:
        skip=set()
        if skip_list is not None: skip|=({skip_list} if not isinstance(skip_list, set) else skip_list)
        if hasattr(model, 'no_weight_decay'): skip|=model.no_weight_decay()
        parameters=get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, verbose=verbose)
        weight_decay=0.
    else: parameters=model.parameters()
    opt_args=dict(lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(parameters, **opt_args)