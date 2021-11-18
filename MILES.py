from typing import List
import numpy as np
from numpy.core.multiarray import concatenate
from numpy.matrixlib import matrix
import util
import sklearn as skl
from sklearn import svm
import argparse
import os
from sklearn.preprocessing import StandardScaler

class MILES():
    def __init__(self, sigma, C):
        self._sigma = sigma
        self._C = C
        self._model = None
        self._concept = None
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, s):
        self._sigma = s

    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self, c):
        self._C = c
        
    @C.setter
    def C(self, c):
        self._C = c
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m

    @property
    def concept(self):
        return self._concept
        
    @concept.setter
    def concept(self, concept):
        self._concept = concept

    def preprocess_data(self, bags_data: np.ndarray, labels: np.ndarray): 
        ''' The embedded feature space requires the matrix to have all positive bags at the front and all negative bags at the end. '''
        pos_data_set = []
        pos_label_set = []
        neg_data_set = []
        neg_label_set = []
        for i in range(len(bags_data)):
            if labels[i] == 0: 
                neg_data_set.append(bags_data[i])
                neg_label_set.append(labels[i])
            else: 
                pos_data_set.append(bags_data[i])
                pos_label_set.append(labels[i])
        bags_matrix = np.concatenate((pos_data_set, neg_data_set), axis=0)
        labels_set = np.concatenate((pos_label_set, neg_label_set), axis=0)
        return [bags_matrix, labels_set]

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
        # print(embedded_bag)
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

    def fit(self, training_data: np.ndarray, training_labels: np.ndarray):
        '''
        function: generate the svm model for prediction purpose
        params: 
            training_data: all the bags for training
            training_labels: the corresponding labels to the bags
        '''
        # bags = np.array([[[-20,-30],[-40,-50]], [[30,30],[20,50]], [[40,40],[10,50]]])
        # labels = np.array([1,0,1])
        # # bags, labels = self.preprocess_data(bags, labels)
        # concepts = np.array([[-20,-30], [-40,-50], [30,30], [20,50], [40,40], [10,50]])
        concept_class = util.generate_concept_class(training_data) # correct number
        # scaler = StandardScaler()
        # scaled_concept_class = scaler.fit_transform(concept_class)
        self.concept = concept_class
        # # print(scaled_concept_class)
        # scaled_training_data = []
        # for bag in training_data:
        #     scaled_bag = scaler.fit_transform(bag)
        #     scaled_training_data.append(scaled_bag)
        # scaled_training_data = np.array(scaled_training_data)
        # print("training data: ", training_data[1][0])
        # print("concept class: ", len(concept_class))
        # i = training_data[1][0] - concept_class[0]
        # print("difference: ", i)
        # print(len(concept_class))
        # print(self.embed_one_bag(concept_class, training_data[0], 3))
        mapped_bags = self.embed_all_bags(concept_class, training_data, self.sigma)
        # mapped_bags = self.embed_all_bags(scaled_concept_class, scaled_training_data, self.sigma)
        # for i in range(len(mapped_bags)):
        #     for j in range(len(mapped_bags[i])):
        #         mapped_bags[i][j] *= 10000000000000000
        svm_l1 = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=self.C, max_iter=100000) # does not support the combination of penalty=l1 and loss=hinge.
        self.model = svm_l1.fit(mapped_bags.T, training_labels)

    def predict(self, test_bags: np.ndarray):
        # embed bags into an instance-based feature space
        concepts = util.generate_concept_class(test_bags)
        mapped_bags = self.embed_all_bags(self.concept, test_bags, self.sigma)
        # print(mapped_bags)
        prediction = self.model.predict(mapped_bags.T)
        return prediction





# def evaluate_and_print_metrics(): 
(bags_train, y_train), (bags_test, y_test) = util.load_musk2()
# bags_train = util.config_irregular_list(bags_train)
y_train = np.array(y_train)
bags_test = util.config_irregular_list(bags_test)
y_test = np.array(y_test)
e = MILES(0.5, 0.3)
bags_train = util.config_irregular_list(bags_train)
y_train = np.array(y_train)
bags_train, y_train = e.preprocess_data(bags_train, y_train)
e.fit(bags_train, y_train)
bags_test, y_test = e.preprocess_data(bags_test, y_test)
prediction = e.predict(bags_test)
print("actual: ", y_test)
def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!
    Args:
        y: True labels.
        y_hat: Predicted labels.
    Returns: Accuracy
    """
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    n = y.size
    return (y == y_hat).sum() / n
print(accuracy(prediction, y_test))
print("prediction: ", prediction)
