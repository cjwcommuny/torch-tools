import torch
from torch import nn

from torchtools.typing import Size


def get_module_output_shape(model: nn.Module, in_shape: Size):
    x = torch.rand(in_shape)
    output = model(x)
    return output.shape

def device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

