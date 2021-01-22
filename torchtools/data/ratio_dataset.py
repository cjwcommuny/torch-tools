import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Dataset, Subset


class RatioSubset(Dataset):
    def __init__(self, dataset: Dataset, ratio: float):
        self.dataset = Subset(
            dataset,
            indices=random.sample(
                population=range(len(dataset)),
                k=int(ratio * len(dataset))
            )
        )
        self.ratio = ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

def ratio_random_split(dataset: Dataset, first_ratio: float, seed: int):
    first_len = int(first_ratio*len(dataset))
    return torch.utils.data.random_split(
        dataset,
        lengths=[first_len, len(dataset) - first_len],
        generator=torch.Generator().manual_seed(seed)
    )