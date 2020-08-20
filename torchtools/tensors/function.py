import itertools
import operator
from functools import reduce

import torch
from torch import Tensor


def value_in_tensor(value: Tensor, tensor: Tensor) -> bool:
    return value.view(1, -1).eq(tensor.view(-1, 1)).sum(0).eq(1).item()

def logical_or(x: Tensor, y: Tensor) -> Tensor:
    return torch.logical_not(torch.logical_not(x) * torch.logical_not(y))

def logical_and(*x) -> Tensor:
    return reduce(operator.mul, x)

def has_nan(x: Tensor) -> bool:
    return torch.isnan(x).any().item()

def has_inf(x: Tensor) -> bool:
    return torch.isinf(x).any().item()

def randbool(*size) -> Tensor:
    return torch.rand(size=size) >= 0.5

def randbool_like(
        input
) -> Tensor:
    return randbool(*input.shape)

def unsqueeze(input: Tensor, dim: int, num: int) -> Tensor:
    new_shape = input.shape[:dim] + (1,) * num + input.shape[dim:]
    return input.reshape(new_shape)


def mask_to_index_1d(mask: Tensor) -> Tensor:
    return torch.nonzero(mask).squeeze()


def if_item_view_1(tensor: Tensor) -> Tensor:
    return tensor.view(1) if tensor.dim() == 0 else tensor