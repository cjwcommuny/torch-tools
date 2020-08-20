import os
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def deterministic():
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    