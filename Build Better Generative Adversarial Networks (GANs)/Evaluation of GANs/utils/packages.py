import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import inception_v3
from torch.distributions import MultivariateNormal
import seaborn as sns # This is for visualization
import sys
import os
torch.manual_seed(0) # Set for our testing purposes, please do not change!
from dotenv import load_dotenv
load_dotenv()