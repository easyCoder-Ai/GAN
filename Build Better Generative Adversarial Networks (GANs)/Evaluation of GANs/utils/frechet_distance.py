from .packages import *
from .matrix_sqrt import matrix_sqrt
def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features) 
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    '''
    #### START CODE HERE ####
    mean_diff = torch.norm(mu_x - mu_y, p=2) ** 2
    sqrt_sigma_x_sigma_y =  matrix_sqrt(sigma_x @ sigma_y)
    trace_term = torch.trace(sigma_x + sigma_y - 2 * sqrt_sigma_x_sigma_y)
    return mean_diff + trace_term
    #### END CODE HERE ####