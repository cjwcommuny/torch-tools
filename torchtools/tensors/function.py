import itertools
import operator
from functools import reduce
from typing import Optional

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
    return torch.nonzero(mask, as_tuple=False).squeeze()

def index2mask_1d(index: Tensor, N: int):
    assert index.ndim == 1
    mask = torch.zeros(N).type_as(index)
    mask.scatter_(dim=0, index=index, value=1)
    return mask

def if_item_view_1(tensor: Tensor) -> Tensor:
    return tensor.view(1) if tensor.dim() == 0 else tensor

def tensor_1d_to_str(arr: Tensor, ndigits: Optional[int]=None) -> str:
    process = lambda x: x if ndigits is None else round(x, ndigits)
    return ','.join([str(process(x.item())) for x in arr])

def str_to_tensor_1d(s: str) -> Tensor:
    lst = [float(x) for x in s.split(',')]
    return torch.tensor(lst)

def tensor_2d_to_str(arr: Tensor, ndigits: Optional[int]=None) -> str:
    return ';'.join([tensor_1d_to_str(arr_1d, ndigits) for arr_1d in arr])

def str_to_tensor_2d(s: str) -> Tensor:
    lst = [[float(x) for x in row.split(',')] for row in s.split(';')]
    return torch.tensor(lst)