from typing import Dict

import torch.nn as nn
from torch import Tensor

from torchtools.modules import ModuleWrapper


class StandardLinear(ModuleWrapper):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float=0.5, bias=True):
        super(StandardLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: Tensor):
        return self.model(x)

    @property
    def weights(self) -> Dict[str, Tensor]:
        return {'linear': self.model[0].weight}

    @property
    def biases(self) -> Dict[str, Tensor]:
        return {'linear': self.model[0].weight}
