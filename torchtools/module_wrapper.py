from abc import ABC, abstractmethod
from typing import Dict, Union

from torch import nn, Tensor

class ModuleWrapper(nn.Module, ABC):
    @property
    @abstractmethod
    def weights(self) -> Dict[str, Tensor]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def biases(self) -> Dict[str, Tensor]:
        raise NotImplementedError()