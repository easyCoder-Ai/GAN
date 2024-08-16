from .packages import *
from .noise import getNoise

class genBlock:

    def __init__(self, inputDim, outputDim) -> None:
        
        self. inputDim = inputDim
        self. outputDim = outputDim


    def getGeneratorBlock(self,):
        '''
        Function for returning a block of the generator's neural network
        given input and output dimensions.
        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar
        Returns:
            a generator neural network layer, with a linear transformation 
            followed by a batch normalization and then a relu activation
        '''
        return nn.Sequential(
            # Hint: Replace all of the "None" with the appropriate dimensions.
            # The documentation may be useful if you're less familiar with PyTorch:
            # https://pytorch.org/docs/stable/nn.html.
            nn.Linear(self. inputDim, self. outputDim),
            nn.BatchNorm1d(self. outputDim),
            nn.ReLU(inplace=True)
        )



class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hiddenDim: the inner dimension, a scalar
    '''
    def __init__(self, zDim=10, imDim=784, hiddenDim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            genBlock(zDim, hiddenDim).getGeneratorBlock(),
            genBlock(hiddenDim, hiddenDim * 2).getGeneratorBlock(),
            genBlock(hiddenDim * 2, hiddenDim * 4).getGeneratorBlock(),
            genBlock(hiddenDim * 4, hiddenDim * 8).getGeneratorBlock(),
            nn.Linear(hiddenDim * 8, imDim),
            nn.Sigmoid())
           
        
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def getGen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen


class genLoss:

    def __init__(self, gen, disc, criterion, numImages, zDim, device) -> None:

        '''
        Return the loss of the generator given inputs.
        Parameters:
            gen: the generator model, which returns an image given z-dimensional noise
            disc: the discriminator model, which returns a single-dimensional prediction of real/fake
            criterion: the loss function, which should be used to compare 
                the discriminator's predictions to the ground truth reality of the images 
                (e.g. fake = 0, real = 1)
            num_images: the number of images the generator should produce, 
                    which is also the length of the real images
            z_dim: the dimension of the noise vector, a scalar
            device: the device type
        Returns:
            gen_loss: a torch scalar loss value for the current batch
        '''

        noises = getNoise(numImages, zDim, device=device)
        fakeImages = gen(noises)
        yP = disc(fakeImages)
        yT = torch.ones(numImages, 1, device=device)
        self. genLoss = criterion(yP, yT)
        
    def getLoss(self):
        
        return self. genLoss