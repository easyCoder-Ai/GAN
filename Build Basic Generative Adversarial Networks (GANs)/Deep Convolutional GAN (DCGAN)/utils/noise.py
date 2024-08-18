from .packages import *

def get_noise(n_samples, zDime, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, zDime)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        zDime: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, zDime, device=device)