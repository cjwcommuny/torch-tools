from typing import Dict, Iterable, Callable

import torch.nn as nn
from torch import Tensor


class FullyConnectedLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_rate: float=0.5,
            activation: Callable=nn.ReLU(),
            bias=True
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            nn.BatchNorm1d(out_features),
            activation,
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: Tensor):
        return self.model(x)


class MultiFullyConnectedLayer(nn.Module):
    def __init__(
            self,
            dims: Iterable[int],
            activation: Callable,
            dropout_rate: float
    ):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])
        self.layers = nn.Sequential(
            *[
                FullyConnectedLayer(in_dim, out_dim, dropout_rate, activation)
                for in_dim, out_dim in dim_pairs
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
