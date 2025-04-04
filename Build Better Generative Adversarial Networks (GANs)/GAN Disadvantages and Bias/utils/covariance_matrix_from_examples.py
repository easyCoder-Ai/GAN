
from .packages import *

def covariance_matrix_from_examples(examples):
    """
    Helper function for get_top_covariances to calculate a covariance matrix. 
    Parameter: examples: a list of steps corresponding to samples of shape (2 * grad_steps, n_images, n_features)
    Returns: the (n_features, n_features) covariance matrix from the examples
    """
    # Hint: np.cov will be useful here - note the rowvar argument!
    ### START CODE HERE ###
    flat_examples = examples.reshape(-1, examples.shape[-1])
    return np.cov(flat_examples, rowvar=False)
    ### END CODE HERE ###