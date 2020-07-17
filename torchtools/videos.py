from typing import Tuple, List

import cv2
from cv2 import VideoCapture
import torch
from torch import Tensor

from torchtools.images import opencv_image_to_torch_tensor


def uniformly_sample(video: VideoCapture, time_segment: int) -> Tuple[Tensor, Tensor]:
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_segment = int(fps * time_segment)

    frame_id = 0
    frame_ids_selected = []
    frames_selected = []
    while True:
        return_val, frame = video.read()
        if frame is None:
            break
        if frame_id % frame_segment == 0:
            frame = opencv_image_to_torch_tensor(frame)
            frames_selected.append(frame)
            frame_ids_selected.append(frame_id)
        frame_id += 1

    frames_selected = torch.stack(frames_selected) # shape=(T, C, H, W)
    frame_ids_selected = torch.tensor(frame_ids_selected)
    assert frames_selected.dim() == 4 and frames_selected.shape[1] == 3
    assert frame_ids_selected.shape == (frames_selected.shape[0],)
    return frame_ids_selected, frames_selected
