from torch import nn
import torch

def weight_decay_parameters_group(net: nn.Module, weight_decay_val: float, skip_list=()):
    """
    exclude bias from weight decay
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: # frozen weights
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name.endswith("bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay_val}]


def l1_norm(parameters):
    return sum([torch.sum(torch.abs(param)) for param in parameters])
