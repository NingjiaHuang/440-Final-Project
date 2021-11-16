import numpy as np
from numpy.core.multiarray import concatenate
import util

class MILES():
    # def __init__(self, sigma):
    #     self._sigma = sigma
    
    # @property
    # def sigma(self):
    #     return self._sigma
    
    def embed_one_bag(self, concept_class: np.ndarray, bag: np.ndarray, sigma: float) -> np.ndarray: 
        '''
        function: embed one bag into the instance feature space
        params: 
            concept_class: all concepts (instances)
            bag: one bag
            sigma: scaling factor
        return:
            an array of the embedded bag. 
        '''
        embedded_bag = []
        for concept in concept_class:
            embedded_bag.append(util.most_likely_cause_estimator(concept, bag, sigma))
        return np.array(embedded_bag)

    def embed_all_bags(self, concept_class: np.ndarray, bags: np.ndarray, sigma: float):
        '''
        function: construct the matrix for all embedded bags
        params: 
            concept_class: all concepts (instances)
            bags: all bags
            sigma: scaling factor
        return:
            an ndarray of the embedded bags.
        '''
        embedded_bags = []
        for bag in bags: 
            embedded_bags.append(self.embed_one_bag(concept_class, bag, sigma))
        return np.array(embedded_bags).T

# bags = np.array([[[2,3],[4,5]], [[3,3],[2,5]], [[4,4],[1,5]]])
# concepts = np.array([[2,3], [4,5], [3,3], [2,5], [4,4], [1,5]])
# print(embed_all_bags(concepts, bags, 0.5))