from torch import nn


class ZeroLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ZeroLayer, self).__init__()

    def forward(self, *args, **kwargs):
        return 0

class Identical(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Empty(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        return args if len(args) > 1 else args[0]


class NoneLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return None
