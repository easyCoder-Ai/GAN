from .packages import *

def remove_from_list(indices, index_to_remove):
    """
    Helper function for get_top_covariances to remove an index from an array. 
    Parameter: indices, a list of indices as a numpy array of shape (n_indices)
    Returns: the numpy array of indices in the same order without index_to_remove
    """
    # Hint: There are many ways to do this, but please don't edit the list in-place.
    # If you're not very familiar with array indexing, you may find this page helpful:
    # https://numpy.org/devdocs/reference/arrays.indexing.html (especially boolean indexing)
    ### START CODE HERE ###
    ### END CODE HERE ###
    return indices[indices != index_to_remove]