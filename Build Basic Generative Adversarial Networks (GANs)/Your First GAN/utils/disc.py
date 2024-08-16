from .packages import *
from .noise import getNoise

class discBlock:

    def __init__(self, inputDim, outputDim) -> None:
        self. inputDim = inputDim
        self. outputDim = outputDim


    def getDiscriminatorBlock(self,):
        '''
        Discriminator Block
        Function for returning a neural network of the discriminator given input and output dimensions.
        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar
        Returns:
            a discriminator neural network layer, with a linear transformation 
            followed by an nn.LeakyReLU activation with negative slope of 0.2 
            (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
        '''
        return nn.Sequential(
            
            nn.Linear(self. inputDim, self. outputDim),
            nn.LeakyReLU(0.2)
            
        )


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, imDim=784, hiddenDim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            discBlock(imDim, hiddenDim * 4).getDiscriminatorBlock(),
            discBlock(hiddenDim * 4, hiddenDim * 2).getDiscriminatorBlock(),
            discBlock(hiddenDim * 2, hiddenDim).getDiscriminatorBlock(),
            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            
            nn.Linear(hiddenDim, 1)
            
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def getDisc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc


class discLoss:

    def __init__(self, gen, disc, criterion, real, numImages, zDim, device) -> None:
        

            '''
            Return the loss of the discriminator given inputs.
            Parameters:
                gen: the generator model, which returns an image given z-dimensional noise
                disc: the discriminator model, which returns a single-dimensional prediction of real/fake
                criterion: the loss function, which should be used to compare 
                    the discriminator's predictions to the ground truth reality of the images 
                    (e.g. fake = 0, real = 1)
                real: a batch of real images
                num_images: the number of images the generator should produce, 
                        which is also the length of the real images
                z_dim: the dimension of the noise vector, a scalar
                device: the device type
            Returns:
                disc_loss: a torch scalar loss value for the current batch
            '''
            noises = getNoise(numImages, zDim, device=device)
            fakeImages = gen(noises).detach() 
            yP = disc(fakeImages)
            yT = torch.zeros(numImages, 1, device=device)
            lossFake = criterion(yP, yT)
            
            yP = disc(real)
            yT = torch.ones(numImages, 1, device=device)
            lossReal = criterion(yP, yT)
            
            self. discLoss = (lossFake + lossReal) / 2
            
         
    def getLoss(self):

        return(self. discLoss)