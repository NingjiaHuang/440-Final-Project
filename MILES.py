from typing import List
import numpy as np
from numpy.core.multiarray import concatenate
from numpy.matrixlib import matrix
import util
import sklearn as skl
from sklearn import svm
import argparse
import os
import os.path
from sklearn.preprocessing import StandardScaler
from Classifier import Classifier
from sklearn import metrics

class MILES(Classifier):
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
    
    def extension_YARDS():
        pass
    
    def fit(self, training_data: np.ndarray, training_labels: np.ndarray):
        '''
        function: generate the svm model for prediction purpose
        params: 
            training_data: all the bags for training
            training_labels: the corresponding labels to the bags
        '''
        concept_class = util.generate_concept_class(training_data) 
        self.concept = concept_class
        mapped_bags = self.embed_all_bags(concept_class, training_data, self.sigma)
        svm_l1 = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=self.C, max_iter=100000) # does not support the combination of penalty=l1 and loss=hinge.
        self.model = svm_l1.fit(mapped_bags.T, training_labels)

    def predict(self, test_bags: np.ndarray):
        # embed bags into an instance-based feature space
        concepts = util.generate_concept_class(test_bags)
        mapped_bags = self.embed_all_bags(self.concept, test_bags, self.sigma)
        prediction = self.model.predict(mapped_bags.T)
        return prediction

def evaluate_and_print_metrics(datapath: str, sigma: float, C: float): 
    (bags_train, y_train), (bags_test, y_test) = util.load_data(datapath)
    bags_train = util.config_irregular_list(bags_train)
    y_train = np.array(y_train)
    bags_test = util.config_irregular_list(bags_test)
    y_test = np.array(y_test)
    e = MILES(sigma, C)
    bags_train, y_train = e.preprocess_data(bags_train, y_train)
    bags_test, y_test = e.preprocess_data(bags_test, y_test)
    e.fit(bags_train, y_train)
    prediction = e.predict(bags_test)
    print("Accuracy Score: ", metrics.accuracy_score(y_test, prediction))
    print("Precision Score: ", metrics.precision_score(y_test, prediction))
    print("Recall Score: ", metrics.recall_score(y_test, prediction))

def miles(datapath: str, sigma: float, C: float):
    evaluate_and_print_metrics(datapath, sigma, C)

if __name__ == '__main__':
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run MILES algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('sigma', type=float, help='Scaling factor.')
    parser.add_argument('C', type=float, help='Penalizing factor.')
    args = parser.parse_args()
    if args.C < 0 or args.C > 1:
       raise argparse.ArgumentTypeError('C must be in range [0,1].')
    path = os.path.expanduser(args.path)
    sigma = args.sigma
    C = args.C
    miles(path, sigma, C)