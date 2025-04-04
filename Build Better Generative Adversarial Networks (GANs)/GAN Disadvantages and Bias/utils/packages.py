import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
torch.manual_seed(0) # Set for our testing purposes, please do not change!
from dotenv import load_dotenv
load_dotenv()