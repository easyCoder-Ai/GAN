
from .packages import *


def get_truncated_noise(n_samples, z_dim, truncation):
    '''
    Generates truncated normal noise for GANs or similar models.
    
    Parameters:
        n_samples (int): Number of samples to generate
        z_dim (int): Dimensionality of each noise vector
        truncation (float): Truncation threshold (in std deviations)
    
    Returns:
        torch.Tensor: Tensor of shape (n_samples, z_dim) with truncated normal noise
    '''
    # Define the bounds in standard deviations
    lower, upper = -truncation, truncation
    
    # Standard normal mean=0, std=1
    truncated_noise = truncnorm.rvs(
        lower, upper, 
        loc=0, scale=1, 
        size=(n_samples, z_dim)
    )
    
    return torch.tensor(truncated_noise, dtype=torch.float32)