import torch
from torch import nn


class ZeroLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ZeroLayer, self).__init__()

    def forward(self, *args, **kwargs):
        return 0
    