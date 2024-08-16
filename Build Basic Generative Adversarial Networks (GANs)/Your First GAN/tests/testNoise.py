
import os
import sys


# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.noise import getNoise
from utils.packages import *

# Verify the noise vector function
def test_get_noise(nSamples, zDim, device='cpu'):
    noise = getNoise(nSamples, zDim, device)
    
    # Make sure a normal distribution was used
    assert tuple(noise.shape) == (nSamples, zDim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)

test_get_noise(1000, 100, 'cpu')
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
print("Success!")