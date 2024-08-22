
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.gradiant import get_gradient
from utils.discriminator import Critic
from utils.packages import *

# UNIT TEST
# DO NOT MODIFY THIS
crit = Critic().to(os.getenv('DEVICE')) 
def test_get_gradient(image_shape):
    real = torch.randn(*image_shape, device=os.getenv('DEVICE')) + 1
    fake = torch.randn(*image_shape, device=os.getenv('DEVICE')) - 1
    epsilon_shape = [1 for _ in image_shape]
    epsilon_shape[0] = image_shape[0]
    epsilon = torch.rand(epsilon_shape, device=os.getenv('DEVICE')).requires_grad_()
    gradient = get_gradient(crit, real, fake, epsilon)
    assert tuple(gradient.shape) == image_shape
    assert gradient.max() > 0
    assert gradient.min() < 0
    return gradient

gradient = test_get_gradient((256, 1, 28, 28))
print("Success!")
