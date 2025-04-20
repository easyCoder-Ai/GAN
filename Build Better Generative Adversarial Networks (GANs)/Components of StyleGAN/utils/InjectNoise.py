
from .packages import *

class InjectNoise(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(  # Use nn.Parameter so weights can be optimized
            torch.randn(1, channels, 1, 1)  # Initialized from a normal distribution
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        # Set the appropriate shape for the noise
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        
        noise = torch.randn(noise_shape, device=image.device)  # Creates the random noise
        return image + self.weight * noise  # Applies to image after multiplying by the weight per channel
    
    #UNIT TEST COMMENT: Required for grading
    def get_weight(self):
        return self.weight
    
    #UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self