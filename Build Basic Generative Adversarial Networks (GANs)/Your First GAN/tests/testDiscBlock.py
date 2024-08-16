
import os
import sys


# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.disc import discBlock
from utils.packages import *

def test_disc_block(inFeatures, outFeatures, numTest=10000):
    block = discBlock(inFeatures, outFeatures).getDiscriminatorBlock()

    # Check there are two parts
    assert len(block) == 2
    test_input = torch.randn(numTest, inFeatures)
    test_output = block(test_input)

    # Check that the shape is right
    assert tuple(test_output.shape) == (numTest, outFeatures)
    
    # Check that the LeakyReLU slope is about 0.2
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5

test_disc_block(25, 12)
test_disc_block(15, 28)
print("Success!")