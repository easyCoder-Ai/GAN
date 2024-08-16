import os
import sys


# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.disc import Discriminator
from utils.packages import *

def testDiscriminator(zDim, hiddenDim, numTest=100):
    
    disc = Discriminator(zDim, hiddenDim).getDisc()

    # Check there are three parts
    assert len(disc) == 4

    # Check the linear layer is correct
    test_input = torch.randn(numTest, zDim)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (numTest, 1)
    
    # Don't use a block
    assert not isinstance(disc[-1], nn.Sequential)

testDiscriminator(5, 10)
testDiscriminator(20, 8)
print("Success!")