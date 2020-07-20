from torch.utils.tensorboard import SummaryWriter
from torch import nn


def log_parameter_distributions(
        tensorboard: SummaryWriter,
        model: nn.Module,
        global_step: int,
        bins: str='tensorflow'
):
    for name, parameter in model.named_parameters():
        name = name.replace('.', '/')
        tensorboard.add_histogram(
            f"param/{name}",
            parameter,
            global_step,
            bins
        )


def log_parameter_grad_distributions(
        tensorboard: SummaryWriter,
        model: nn.Module,
        global_step: int,
        bins: str='tensorflow'
):
    for name, parameter in model.named_parameters():
        name = name.replace('.', '/')
        tensorboard.add_histogram(
            f"grad/{name}",
            parameter.grad,
            global_step,
            bins
        )
