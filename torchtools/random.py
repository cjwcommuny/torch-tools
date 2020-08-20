import torch
from torch import Tensor


def choice(low: int, high: int, num: int, device=None) -> Tensor:
    """
    random choose `num` elements from [low, high)
    :param low:
    :param high:
    :param num:
    :param device:
    :return:
    """
    return low + torch.randperm(high - low, device=device)[:num]
