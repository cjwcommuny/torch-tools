import json
import os
from typing import Callable, Optional

from torch.utils.data import Dataset


class PredefinedDatasetFolder(Dataset):
    def __init__(
            self,
            root: str,
            index: str,
            loader: Callable,
            transform: Optional[Callable]=None,
            target_transform: Optional[Callable]=None,
            return_index: bool=False
    ):
        """
        :param root:
        :param index: json file format like [(folder_name: str, class: str)]
        :param loader:
        """
        super().__init__()
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index
        index = json.load(open(index, 'r'))
        self.classes = sorted(list(set(cls for folder_name, cls in index)))
        self.class2index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = [
            (os.path.join(root, cls, folder_name), self.class2index[cls])
            for folder_name, cls in index
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return index, sample, target
        else:
            return sample, target