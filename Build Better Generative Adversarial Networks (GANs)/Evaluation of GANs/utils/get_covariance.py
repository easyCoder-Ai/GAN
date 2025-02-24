import numpy as np
from .packages import *
def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))