
from .packages import *


def showTensorImages(imageTensor, numImages=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    imageTensor = (imageTensor + 1) / 2
    imageUnflat = imageTensor.detach().cpu()
    imageGrid = make_grid(imageUnflat[:numImages], nrow=5)
    plt.imshow(imageGrid.permute(1, 2, 0).squeeze())
    plt.show()

    