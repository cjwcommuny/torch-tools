from torch import nn

class GeneralSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args):
        for layer in self.layers:
            args = layer(*args)
        return args

