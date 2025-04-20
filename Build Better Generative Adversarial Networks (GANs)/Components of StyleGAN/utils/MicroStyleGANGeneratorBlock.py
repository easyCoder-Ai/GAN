from .packages import *
from .InjectNoise import InjectNoise
from .AdaIN import AdaIN


class MicroStyleGANGeneratorBlock(nn.Module):
    '''
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    '''

    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample

        # 1. Upsample to the starting_size using bilinear interpolation
        if self.use_upsample:
            self.upsample = nn.Upsample(size=starting_size, mode='bilinear', align_corners=False)

        # 2. Convolution layer
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1)  # assumes kernel_size=3

        # 3. InjectNoise object
        self.inject_noise = InjectNoise(out_chan)

        # 4. AdaIN object
        self.adain = AdaIN(out_chan, w_dim)

        # 5. Activation function
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        '''
        Forward pass of MicroStyleGANGeneratorBlock
        Parameters:
            x: feature map of shape (n_samples, channels, width, height)
            w: intermediate noise vector
        '''
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.adain(x, w)
        x = self.activation(x)
        return x

    #UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self