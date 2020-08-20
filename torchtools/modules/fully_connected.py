from typing import Iterable, Callable, Optional, List

import torch.nn as nn
from torch import Tensor

from torchtools.modules.placeholder import Identical


class FullyConnectedLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_rate: float=0.5,
            activation: nn.Module=nn.ReLU(),
            norm_layer: Optional[str]='batchnorm',
            bias=True
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = Identical()
        if norm_layer == 'batchnorm':
            norm_layer = nn.BatchNorm1d(out_features)
        elif norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm(out_features)
        elif norm_layer == 'groupnorm':
            norm_layer = nn.GroupNorm(num_groups=32, num_channels=out_features)
        else:
            raise ValueError(f'norm_layer should not be {norm_layer}')
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            norm_layer,
            activation,
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: Tensor):
        return self.model(x)


class MultiFullyConnectedLayer(nn.Module):
    def __init__(
            self,
            dims: List[int],
            dropout_rate: float,
            norm_layer: Optional[str]='batchnorm',
            activation: nn.Module=nn.ReLU()
    ):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])
        self.layers = nn.Sequential(
            *[
                FullyConnectedLayer(in_dim, out_dim, dropout_rate, activation, norm_layer)
                for in_dim, out_dim in dim_pairs
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
