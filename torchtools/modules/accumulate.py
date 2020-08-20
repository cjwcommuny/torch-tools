import itertools
from typing import Iterable

from torch import nn

class Accumulate(nn.Sequential):
    def forward(self, inputs: Iterable, init=None) -> list:
        return list(itertools.accumulate(inputs, func=self._modules.values(), initial=init))

