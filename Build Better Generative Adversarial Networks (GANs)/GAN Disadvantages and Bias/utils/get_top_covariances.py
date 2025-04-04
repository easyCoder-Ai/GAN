from .packages import *


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CELL: get_top_covariances
def get_top_covariances(classification_changes, target_index, top_n=10):
    '''
    Returns the indices of the features with the highest covariance (absolute value)
    with the target_index feature.
    '''
    # Flatten (steps * images) into one dimension
    flat = classification_changes.reshape(-1, classification_changes.shape[-1])
    
    # Step 1: Compute covariance matrix
    cov_matrix = np.cov(flat, rowvar=False)  # shape: (n_features, n_features)

    # Step 2: Extract covariances for the target feature
    covariances_with_target = cov_matrix[target_index]

    # Step 3: Remove the target index from consideration
    all_indices = np.arange(covariances_with_target.shape[0])
    remaining_indices = all_indices[all_indices != target_index]
    remaining_covariances = covariances_with_target[remaining_indices]

    # Step 4: Get top N by magnitude
    sorted_indices = np.argsort(np.abs(remaining_covariances))[::-1][:top_n]
    relevant_indices = remaining_indices[sorted_indices]
    highest_covariances = remaining_covariances[sorted_indices]

    return relevant_indices, highest_covariances