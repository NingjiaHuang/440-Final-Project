import numpy as np
from typing import List

def most_likely_cause_estimator(concept_instance: np.array, bag: np.ndarray, sigma: float) -> float:
    '''
    function: similarity measure between the bag and the concept
    params: 
        concept_instance: an instance in the dataset 
        bag: bag to be classified
        sigma: scaling factor
    return: 
        maximum similarity between the target concept and the bag instances
    '''
    similarity_list = []
    for instance in bag: 
        similarity_list.append(np.exp(-(np.linalg.norm(instance - concept_instance) ** 2/(sigma ** 2))))
    return max(similarity_list)