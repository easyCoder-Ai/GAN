from .packages import *
from .MicroStyleGANGeneratorBlock import MicroStyleGANGeneratorBlock
from .MappingLayers import MappingLayers
class MicroStyleGANGenerator(nn.Module):
    '''
    Micro StyleGAN Generator Class
    '''

    def __init__(self, 
                 z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)

        # Starting learned constant input
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))

        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)

        # 1x1 Convs to output RGB images
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)

        # Alpha for smooth fade-in interpolation
        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        '''
        Upsamples the first image to match the size of the second image
        '''
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):
        '''
        Forward pass for the generator
        '''
        x = self.starting_constant
        w = self.map(noise)

        x = self.block0(x, w)
        x_small = self.block1(x, w)
        x_small_image = self.block1_to_image(x_small)

        x_big = self.block2(x_small, w)
        x_big_image = self.block2_to_image(x_big)

        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image)

        # Alpha-blended interpolation between two resolution outputs
        interpolation = (1 - self.alpha) * x_small_upsample + self.alpha * x_big_image

        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation

    #UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self