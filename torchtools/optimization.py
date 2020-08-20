import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def weight_decay_parameters_group(net: nn.Module, weight_decay_val, skip_list=()):
    """
    exclude bias from weight decay
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: # frozen weights
            continue
        if len(param.shape) == 1 or name.endswith("bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay_val}]


def l1_norm(parameters):
    return sum([torch.sum(torch.abs(param)) for param in parameters])


def get_linear_schedule_with_warmup(optimizer, max_lr: float, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    :param optimizer:
    :param max_lr:
    :param num_warmup_steps:
    :param num_training_steps:
    :param last_epoch:
    :return:
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) * max_lr / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) * max_lr / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
