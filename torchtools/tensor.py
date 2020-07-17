import torch
from torch import Tensor

def value_in_tensor(value: Tensor, tensor: Tensor) -> bool:
    return value.view(1, -1).eq(tensor.view(-1, 1)).sum(0).eq(1).item()


def logical_or(x: Tensor, y: Tensor):
    return torch.logical_not(torch.logical_not(x) * torch.logical_not(y))

