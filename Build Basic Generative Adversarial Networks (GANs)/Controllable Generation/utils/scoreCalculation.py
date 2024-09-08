from .packages import *

def get_score(current_classifications, original_classifications, target_indices, other_indices, penalty_weight):
    '''
    Function to return the score of the current classifications, penalizing changes
    to other classes with an L2 norm.
    Parameters:
        current_classifications: the classifications associated with the current noise
        original_classifications: the classifications associated with the original noise
        target_indices: the index of the target class
        other_indices: the indices of the other classes
        penalty_weight: the amount that the penalty should be weighted in the overall score
    '''
    #### START CODE HERE ####
    # Calculate the error on other_indices    
    other_distances = original_classifications[:, other_indices] - current_classifications[:, other_indices]
    # Calculate the norm (magnitude) of changes per example and multiply by penalty weight  
    other_class_penalty = -penalty_weight * torch.mean(torch.norm(other_distances, dim=1))
    # Take the mean of the current classifications for the target feature  
    target_score = torch.mean(current_classifications[:, target_indices])
    #### END CODE HERE ####
    return target_score + other_class_penalty
