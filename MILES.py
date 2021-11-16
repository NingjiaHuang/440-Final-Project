import numpy as np
import util

class MILES():
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

        
# bag1 = np.matrix([[2,3],[4,5]])
# concept1 = np.matrix([[2,3],[4,5],[2,2],[3,4]])
# res = np.array([1.00000000e+00,1.00000000e+00,1.83156389e-02,3.35462628e-04])
# print(embed_one_bag(concept1, bag1, 0.5))
# print(embed_one_bag(concept1, bag1, 0.5))
# print(res)
# print(np.array_equal(res, embed_one_bag(concept1, bag1, 0.5)))
# print(embed_one_bag(concept1, bag1, 0.5)[2]-res[2])