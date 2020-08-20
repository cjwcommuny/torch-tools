from typing import Tuple

import torch
from torch import Tensor
from torch import nn


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, dim_encoding: int):
        super(AbsolutePositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_length, dim_encoding)

    def forward(self, length: int) -> Tensor:
        return self.embedding(torch.arange(length).to(self.embedding.weight.device))



class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, dim_encoding: int):
        super(RelativePositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_length, dim_encoding)

    def forward(self, length: int, centers: Tuple[int, int]) -> Tensor:
        """
        :param length:
        :param centers: [start, end)
        :return:
        """
        start, end = centers
        before = torch.arange(start).to(self.embedding.weight.device)
        after = torch.arange(end, length).type_as(before)
        before: Tensor = start - before
        after: Tensor = after - end + 1
        context = torch.cat((before, after))
        assert context.shape == (length - (end - start),)
        return self.embedding(context)

