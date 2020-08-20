from torch import Tensor, nn

class Parallel(nn.Sequential):
    def forward(self, *args, **kwargs):
        return [module(*args, **kwargs) for module in self._modules.values()]

