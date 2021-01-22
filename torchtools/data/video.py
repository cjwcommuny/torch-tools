from typing import List

import torch
from torch import Tensor


def normalize_batched_video(tensor: Tensor, mean: List[float], std: List[float], inplace: bool=False):
    """
    :param tensor: shape=(batch_size, T, C, H, W)
    :param mean:
    :param std:
    :param inplace:
    """
    assert tensor.ndim == 5
    if not inplace:
        tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(tensor.dtype)
        )
    if mean.ndim == 1:
        mean = mean[None, None, :, None, None]
    if std.ndim == 1:
        std = std[None, None, :, None, None]
    tensor.sub_(mean).div_(std)
    return tensor