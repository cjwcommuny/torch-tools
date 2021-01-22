from torch import nn
from torch import Tensor

class Expand(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor):
        return x.expand(*self.shape)

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor):
        return x.view(*self.shape)

class Rescale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x
