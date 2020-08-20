from typing import Tuple, Callable

import torch
from torch import Tensor


def compute_pairwise_accuracy(
        tensor1: Tensor,
        tensor2: Tensor,
        is_correct: Callable=lambda x, y: x > y
) -> Tuple[int, int]:
    pairs = torch.cartesian_prod(tensor1.view(-1), tensor2.view(-1))
    bingo = sum([is_correct(p[0], p[1]) for p in pairs])
    if isinstance(bingo, Tensor):
        bingo = bingo.item()
    all = len(pairs)
    return bingo, all
