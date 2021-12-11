from typing import _SpecialForm, List
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression

class MILES(Classifier):
    def __init__(self, sigma, C):
        self._sigma = sigma
        self._C = C
        self._model = None
        self._concept = None
        self._model_tree = None
        self._model_logistic = None
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
    
    @property
    def model_tree(self):
        return self._model_tree
        
    @model_tree.setter
    def model_tree(self, model_tree):
        self._model_tree = model_tree

    @property
    def model_logistic(self):
        return self._model_logistic
        
    @model_logistic.setter
    def model_logistic(self, model_logistic):
        self._model_logistic = model_logistic

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
            embedded_bag.append(util.sum_likelihood_estimator(concept, bag, sigma))
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
        concept_class = util.generate_concept_class(training_data) 
        self.concept = concept_class
        mapped_bags = self.embed_all_bags(concept_class, training_data, self.sigma)
        svm_l1 = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=self.C, max_iter=100000) # does not support the combination of penalty=l1 and loss=hinge.
        self.model = svm_l1.fit(mapped_bags.T, training_labels)
        '''----------------------Extension----------------------'''
        self.model_tree = tree.DecisionTreeClassifier().fit(mapped_bags.T, training_labels)
        self.model_logistic = LogisticRegression(random_state=0).fit(mapped_bags.T, training_labels)

    def predict(self, test_bags: np.ndarray):
        # embed bags into an instance-based feature space
        concepts = util.generate_concept_class(test_bags)
        mapped_bags = self.embed_all_bags(self.concept, test_bags, self.sigma)
        prediction_logistic = self.model_logistic.predict(mapped_bags.T)
        prediction_tree = self.model_tree.predict(mapped_bags.T)
        prediction_svm = self.model.predict(mapped_bags.T)
        return majority_vote([prediction_logistic, prediction_svm, prediction_tree])

def majority_vote(list_of_prediction: np.ndarray) -> np.ndarray:
        prediction = []
        for i in range(0, len(list_of_prediction[0])):
            temp_list = []
            for element in list_of_prediction:
                temp_list.append(element[i])
                maj_vote = np.bincount(temp_list).argmax()
            prediction.append(maj_vote)
        return np.array(prediction)

def evaluate_and_print_metrics(datapath: str, sigma: float, C: float): 
    (bags_train, y_train), (bags_test, y_test) = util.load_data(datapath)
    bags = []
    bags.extend(bags_train)
    bags.extend(bags_test)
    y = []
    y.extend(y_train)
    y.extend(y_test)
    e = MILES(sigma, C)
    acc_list, rec_list, pre_list, auc_list = [], [], [], []
    for count in range(0, 10): 
        bags_train, bags_test, y_train, y_test = train_test_split(bags, y, test_size=0.1)
        bags_train = util.config_irregular_list(bags_train)
        y_train = np.array(y_train)
        bags_test = util.config_irregular_list(bags_test)
        y_test = np.array(y_test)
        bags_train, y_train = e.preprocess_data(bags_train, y_train)
        bags_test, y_test = e.preprocess_data(bags_test, y_test)
        e.fit(bags_train, y_train)
        prediction = e.predict(bags_test)
        # res = majority_vote([prediction, prediction2, prediction3])
        # acc_list.append(metrics.accuracy_score(y_test, prediction))
        # rec_list.append(metrics.recall_score(y_test, prediction))
        # pre_list.append(metrics.precision_score(y_test, prediction))
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction)
        # auc_list.append(metrics.auc(fpr, tpr))
        acc_list.append(metrics.accuracy_score(y_test, prediction))
        rec_list.append(metrics.recall_score(y_test, prediction))
        pre_list.append(metrics.precision_score(y_test, prediction))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction)
        auc_list.append(metrics.auc(fpr, tpr))
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    prec_mean, prec_std = np.mean(pre_list), np.std(pre_list)
    recall_mean, recall_std = np.mean(rec_list), np.std(rec_list)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
    print("Accuracy:", np.round(acc_mean, 3),  np.round(acc_std, 3))
    print("Precision:", np.round(prec_mean, 3),  np.round(prec_std, 3))
    print("Recall:", np.round(recall_mean, 3),  np.round(recall_std, 3))
    print("AUC:", np.round(auc_mean, 3), np.round(auc_std, 3))

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