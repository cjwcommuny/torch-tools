from typing import Tuple, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def batch_to_cuda(batch: Tuple[Tensor], device='cuda:0'):
    return tuple(x.to(device) for x in batch)


def pad_batch(sequence: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    :param sequence:
    :return:
        - sequence_padded: Tensor
        - len_of_tensor: LongTensor
    """
    lens = [len(tensor) for tensor in sequence]
    lens = torch.tensor(lens, dtype=torch.long).to(sequence[0].device)
    sequence_padded: Tensor = pad_sequence(sequence, batch_first=True)
    assert len(lens) == len(sequence_padded)
    return sequence_padded, lens


def unpad_batch(sequence: Tensor, batch_lengths: Tensor) -> List[Tensor]:
    """
    :param sequence: shape=(batch_size, *)
    :param batch_lengths: shape=(batch_size,)
    :return:
        - sequence: len=batch_size
    """
    return [tensor[:tensor_len] for tensor, tensor_len in zip(sequence.unbind(), batch_lengths)]