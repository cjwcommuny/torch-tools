from typing import Tuple, Union, List, Dict

from torch import Tensor

from torchtools.tensors import mask_to_index_1d


class TensorIndex:
    def __init__(self, indexes: List[Tuple[str, Tensor]]):
        """
        :param indexes: index should be 1-D
        """
        self.indexes = indexes
        self.levels: Dict[str, int] = {name: idx for idx, (name, _) in enumerate(indexes)}

    def index_select(self, source: str, destination: str, idx: Union[int, Tensor]) -> Tensor:
        start, stop = self.levels[source], self.levels[destination]
        result = idx
        for i in range(start, stop):
            map = self.indexes[i]
            result = map[result]
        return result

    def mask_select(self, source: str, destination: str, mask: Tensor) -> Tensor:
        idx = mask_to_index_1d(mask)
        return self.index_select(source, destination, idx)
