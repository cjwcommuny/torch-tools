import numpy as np
import torchvision.transforms.functional as torchvision_fn
from PIL import Image
from torch import Tensor


def opencv_image_to_torch_tensor(image: np.ndarray) -> Tensor:
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return torchvision_fn.to_tensor(image)
