
import os
import sys


# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.gen import Generator
from utils.packages import *



# Verify the generator class
def testGenerator(zDim, imDim, hiddenDim, numTest=10000):

    gen = Generator(zDim, imDim, hiddenDim).getGen()
    
    # Check there are six modules in the sequential part
    assert len(gen) == 6
    testInput = torch.randn(numTest, zDim)
    testOutput = gen(testInput)

    # Check that the output shape is correct
    assert tuple(testOutput.shape) == (numTest, imDim)
    assert testOutput.max() < 1, "Make sure to use a sigmoid"
    assert testOutput.min() > 0, "Make sure to use a sigmoid"
    assert testOutput.min() < 0.5, "Don't use a block in your solution"
    assert testOutput.std() > 0.05, "Don't use batchnorm here"
    assert testOutput.std() < 0.15, "Don't use batchnorm here"

testGenerator(5, 10, 20)
testGenerator(20, 8, 24)
print("Success!")