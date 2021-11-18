import numpy as np
from typing import List
from mil.data.datasets.loader import load_data

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
        if len(np.unique(instance)) > 1: 
            similarity_list.append(((np.linalg.norm(instance - concept_instance) ** 2/(sigma ** 2))))
    return max(similarity_list)

def generate_concept_class(bags: np.ndarray) -> np.ndarray:
    '''Decompose each bag to make all instances into an ndarray.'''
    concept_class = []
    for bag in bags:
        for example in bag:
            if len(np.unique(example)) != 1:
                concept_class.append(np.array(example))
    # print(type(np.array(concept_class)))
    return np.array(concept_class)

def config_irregular_list(raw_list: List) -> np.ndarray:
    '''
    function: convert list of lists with irregular length into ndarray
    '''
    res_array = np.zeros([len(raw_list),len(max(raw_list, key = lambda x: len(x))), len(raw_list[0][0])])
    # res_array = np.zeros([len(raw_list),40, len(raw_list[0][0])])
    for i in range(len(raw_list)):
        for j in range(len(max(raw_list, key = lambda x: len(x)))):
        # for j in range(40):
            for k in range(len(raw_list[0][0])):
                res_array[i][j][k] = 100000000
    for i,j in enumerate(raw_list):
        res_array[i][0:len(j)] = j
    return res_array

def load_data_csv(path):
    return load_data(path)