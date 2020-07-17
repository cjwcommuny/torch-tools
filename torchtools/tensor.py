from typing import Tuple

import torch
from torch import Tensor

def value_in_tensor(value: Tensor, tensor: Tensor) -> bool:
    return value.view(1, -1).eq(tensor.view(-1, 1)).sum(0).eq(1).item()

def logical_or(x: Tensor, y: Tensor):
    return torch.logical_not(torch.logical_not(x) * torch.logical_not(y))

def has_nan(x: Tensor) -> bool:
    return torch.isnan(x).any().item()

def has_inf(x: Tensor) -> bool:
    return torch.isinf(x).any().item()
