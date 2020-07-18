from typing import Dict, Tuple

import torch
from torch import Tensor

from torchtools.tensors.function import unsqueeze


class TensorGroup:
    def __init__(self, tensors: Dict[str, Tensor], check_validity: bool=True):
        self.tensors = tensors
        if check_validity:
            assert self.check_tensors_len(), "tensors don't has same length"


    def check_tensors_len(self) -> bool:
        return len(set(map(lambda x: len(x), self.tensors.values()))) == 1

    @property
    def columns(self):
        return self.tensors.keys()

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.tensors[item]
        else:
            return TensorGroup(
                {key: tensor[item] for key, tensor in self.tensors.items()},
                check_validity=False
            )

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.tensors[key] = value
        else:
            for k in self.tensors.keys():
                self.tensors[k][key] = value


    def min(self, input: str) -> Tuple['TensorGroup', Tensor]:
        indices = self.tensors[input].argmin(dim=0, keepdim=True)
        return (
            TensorGroup(
                {key: tensor[indices] for key, tensor in self.tensors.items()},
                check_validity=False
            ),
            indices
        )

    def max(self, input: str) -> Tuple['TensorGroup', Tensor]:
        indices = self.tensors[input].argmax(dim=0, keepdim=True)
        return (
            TensorGroup(
                {key: tensor[indices] for key, tensor in self.tensors.items()},
                check_validity=False
            ),
            indices
        )


    def sort(self, input: str, descending: bool=False) -> Tuple['TensorGroup', Tensor]:
        indices = self.tensors[input].argsort(dim=0, descending=descending)
        return (
            TensorGroup(
                {key: tensor[indices] for key, tensor in self.tensors.items()},
                check_validity=False
            ),
            indices
        )

    def topk(self, input: str, k: int, largest: bool=True, sorted: bool=True) -> Tuple['TensorGroup', Tensor]:
        _, indices = self.tensors[input].topk(k, largest=largest, sorted=sorted)
        return (
            TensorGroup(
                {key: tensor[indices] for key, tensor in self.tensors.items()},
                check_validity=False
            ),
            indices
        )


    @staticmethod
    def __mask_reshape_to_broadcast(tensor: Tensor, mask: Tensor) -> Tensor:
        result = unsqueeze(mask, dim=1, num=tensor.dim() - 1)
        print(f"mask.shape: {result.shape}")
        return result

    def masked_select(self, mask: Tensor) -> 'TensorGroup':
        assert mask.dim() == 1
        print(f"mask: {mask.shape}")
        print(f"features: {self.tensors['features'].shape}")
        print(f"frame_id: {self.tensors['frame_id'].shape}")
        return TensorGroup(
            {key: tensor.masked_select(self.__mask_reshape_to_broadcast(tensor, mask)) for key, tensor in self.tensors.items()},
            check_validity=False
        )

    def index_select(self, indices: Tensor) -> 'TensorGroup':
        assert indices.dim() == 1
        return self[indices]

    def lt_select(self, input: str, other) -> 'TensorGroup':
        """
        NOTE: tensor.dim() must all be 1
        """
        mask = torch.lt(self.tensors[input], other)
        return self.masked_select(mask)

    def le_select(self, input: str, other) -> 'TensorGroup':
        """
        NOTE: tensor.dim() must all be 1
        """
        mask = torch.le(self.tensors[input], other)
        return self.masked_select(mask)

    def gt_select(self, input: str, other) -> 'TensorGroup':
        """
        NOTE: tensor.dim() must all be 1
        """
        mask = torch.gt(self.tensors[input], other)
        return self.masked_select(mask)

    def ge_select(self, input: str, other) -> 'TensorGroup':
        """
        NOTE: tensor.dim() must all be 1
        """
        mask = torch.ge(self.tensors[input], other)
        return self.masked_select(mask)

    def ne_select(self, input: str, other) -> 'TensorGroup':
        """
        NOTE: tensor.dim() must all be 1
        """
        mask = torch.ne(self.tensors[input], other)
        return self.masked_select(mask)

    def eq_select(self, input: str, other) -> 'TensorGroup':
        """
        NOTE: tensor.dim() must all be 1
        """
        mask = torch.eq(self.tensors[input], other)
        return self.masked_select(mask)