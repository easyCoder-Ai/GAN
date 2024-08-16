from .packages import *
class imageShow:

    def __init__(self, imageTensor, numImages=25, size=(1, 28, 28)) -> None:
        
        self. imageTensor = imageTensor
        self. numImages = numImages
        self. size = size


    def showTensorImages(self,):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in a uniform grid.
        '''
        imageUnflat = self. imageTensor.detach().cpu().view(-1, * self. size)
        imageGrid = make_grid(imageUnflat[:self. numImages], nrow=5)
        plt.imshow(imageGrid.permute(1, 2, 0).squeeze())
        plt.show()