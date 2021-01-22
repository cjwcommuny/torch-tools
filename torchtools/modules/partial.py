from torch import nn
from functools import partial

class Partial(nn.Module):
    def __init__(self, module: nn.Module, *args, **kwargs):
        super().__init__()
        self.module = module
        self.partial = partial(module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.partial(*args, **kwargs)
