import torchvision.transforms.functional as torchvision_fn
import numpy as np
from torch import Tensor
from PIL import Image

def opencv_image_to_torch_tensor(image: np.ndarray) -> Tensor:
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return torchvision_fn.to_tensor(image)
