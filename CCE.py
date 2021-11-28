from collections import defaultdict
from typing import List
import numpy as np
import util
from sklearn import svm
import argparse
import os
import os.path
from Classifier import Classifier
from sklearn.cluster import KMeans
from sklearn import metrics
from mil.data.datasets.loader import load_data
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

class KMEANS:
    def __init__(self, k: int, max_iter=2000):
        self._k = k
        self._max_iter = max_iter
        self._centroids = None
        self._clusters = None
    
    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        self._k = k

    @property
    def max_iter(self):
        return self._max_iter
    
    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self, centroids):
        self._centroids = centroids

    @property
    def clusters(self):
        return self._clusters
    
    @clusters.setter
    def clusters(self, clusters):
        self._clusters = clusters

    def init_centroids(self, X: np.ndarray): 
        '''Randomly choose k centroids for initialization.'''
        num_instances, num_features = np.shape(X)
        rand_center = np.random.choice(num_instances, self.k, replace=False) # sample without replacement
        centroids = X[rand_center, :].copy()
        self.centroids = np.array(centroids)
        return self.centroids
    
    def update_assignment(self, X: np.ndarray): 
        '''Assign each data point to the closest centroid.'''
        clusters = []
        for instance in X:
            clusters.append(np.argmin(util.euclidean_distance(instance, self.centroids)))
        self.clusters = np.array(clusters)
        return self.clusters

    def shift_centroids(self, X: np.ndarray):
        '''Compute and place the new centroid of each cluster.'''
        new_centroids = []
        for cluster in range(0, self.k):
            index = np.where(self.clusters == cluster)
            cluster_set = X[index]
            new_centroids.append(np.mean(cluster_set, axis=0))
        self.centroids = np.array(new_centroids)
        return self.centroids
        
    def fit(self, X: np.ndarray):
        cluster_success = False
        centroids = self.init_centroids(X)
        for i in range(0, self.max_iter):
            clusters = self.update_assignment(X)
            prev_centroids = centroids.copy()
            curr_centroids = self.shift_centroids(X)
            difference = curr_centroids - prev_centroids
            if not difference.any():
                cluster_success = True
                return cluster_success
        self.centroids = centroids
        self.clusters = clusters
        return cluster_success
    
    def predict(self, X):
        y_preds = self.update_assignment(self.centroids, X)
        return y_preds

class CCE(Classifier):
    def __init__(self, d: int, max_iter: int):
        self._d = d
        self._max_iter = max_iter
        self._models = None

    @property
    def d(self):
        return self._d
    
    @d.setter
    def d(self, d):
        self._d = d

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter
    
    @property
    def models(self):
        return self._models
    
    @models.setter
    def models(self, models):
        self._models = models

    def overlap(self, bag: np.ndarray, group_k: np.ndarray) -> int:
        '''
        function: find whether there are common instances between a bag and a cluster. 
        params: 
            bag: a bag of instances.
            group_k: a cluster of instances.
        return: 
            1 if there are common instances and 0 otherwise. 
        '''
        for instance in bag:
            for instance_in_cluster in group_k:
                if np.array_equal(instance, instance_in_cluster):
                    return 100
        return 0

    def select_by_index(self, cluster_index: List, num_of_cluster: int, concept_class: np.ndarray) -> List:
        '''
        function: a helper method that selects all the elements with same indexes (clustered by kmeans) to form a set.
        params: 
            cluster_index: a List of indexes of clusters.
            num_of_cluster: the total number of clusters.
            concept_class: the unpacked bags (that is, a collection of instances).
        return: 
            A list of clusters (each cluster contains the corresponding instances). 
        '''
        clusters = []
        for i in range(0, num_of_cluster):
            cluster = []
            index = np.where(cluster_index == i)
            cluster = concept_class[index]
            clusters.append(cluster)
        return clusters

    def generate_bag_feature_vector(self, concept_class: np.ndarray, len_bags: int, X: List) -> List: 
        '''
        function: generate the binary representation of each bag based on whether it contains instances from the kth cluster. 
        params: 
            concept_class: all concepts (instances).
            len_bags: the number of bags. 
            X: the list of bags.
        return:
            A list of new representation of features of each bag. 
        '''
        X = util.config_irregular_list(X)
        config_feature_rep = []
        for i in self.d:

            # using KMEANS 
            kmeans = KMEANS(i, self.max_iter)
            kmeans.fit(concept_class)
            clustering_result = kmeans.clusters

            # using KPCA
            # kpca = KernelPCA(n_components=None, kernel='precomputed')
            # lambda_kpca = 0.6
            # kernel_old = lambda_kpca * pow(np.dot(concept_class, concept_class.T), 2) + (1-lambda_kpca)*rbf_kernel(concept_class)
            # old_kpca = kpca.fit_transform(kernel_old)
            # kmeans = KMeans(n_clusters=self.d[0], random_state=0).fit(old_kpca)
            # clustering_result = kmeans.labels_
            clusters = self.select_by_index(clustering_result, i, concept_class)
            s_i = []
            for j in range(0, len_bags):
                bag_vector = []
                for k in clusters:
                    k = np.array(k)
                    instances_in_bag = np.array(X[j])
                    bag_vector.append(self.overlap(instances_in_bag, k))
                s_i.append(np.array(bag_vector))
            s_i = np.array(s_i)
            config_feature_rep.append(s_i)
        return config_feature_rep
    
    def majority_vote(self, list_of_prediction: np.ndarray) -> np.ndarray:
        prediction = []
        for i in range(0, len(list_of_prediction[0])):
            temp_list = []
            for element in list_of_prediction:
                temp_list.append(element[i])
                maj_vote = np.bincount(temp_list).argmax()
            prediction.append(maj_vote)
        return np.array(prediction)
                
    def fit(self, concept_class: np.ndarray, len_bags: int, X: List, labels: np.ndarray):
        config_feature_rep = self.generate_bag_feature_vector(concept_class, len_bags, X)
        models = []
        for feature_rep in config_feature_rep:
            model = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=0.5, max_iter=10000).fit(feature_rep, labels)
            models.append(model)
        self.models = models

    def predict(self, test_concept_class: np.ndarray, len_bags: int, bags_test: List):
        models = self.models
        config_feature_rep = self.generate_bag_feature_vector(test_concept_class, len_bags, bags_test)
        prediction_list = []
        for i in range(len(models)): 
            prediction = models[i].predict(config_feature_rep[i])
            prediction_list.append(prediction)
        ret_prediction = self.majority_vote(prediction_list)
        return ret_prediction

def evaluate_and_print_metrics(datapath: str, d: int, max_iter: int): 
    (bags_train, y_train), (bags_test, y_test) = util.load_data(datapath)
    list_bag_train = bags_train.copy()
    list_bag_test = bags_test.copy()
    bags_train = util.config_irregular_list(bags_train)
    concept_class = util.generate_concept_class(bags_train)
    y_train = np.array(y_train)
    bags_test = util.config_irregular_list(bags_test)
    y_test = np.array(y_test)
    test_concept_class = util.generate_concept_class(bags_test)
    e = CCE(d, max_iter)
    e.fit(concept_class, len(list_bag_train), list_bag_train, y_train)
    prediction = e.predict(test_concept_class, len(list_bag_test), list_bag_test)
    print("Accuracy Score: ", metrics.accuracy_score(y_test, prediction))
    print("Precision Score: ", metrics.precision_score(y_test, prediction))
    print("Recall Score: ", metrics.recall_score(y_test, prediction))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction)
    print("AUC Score: ", metrics.auc(fpr, tpr))

def cce(datapath: str, d: int, max_iter: int):
    evaluate_and_print_metrics(datapath, d, max_iter)
    
if __name__ == '__main__':
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run CCE algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('-d', type=int, nargs="+", help='The number of clusters for the clustering algorithm.')
    parser.add_argument('-max_iter', type=int, help='The maximum number of iteration for clustering algorithm.')
    args = parser.parse_args()
    path = os.path.expanduser(args.path)
    d = args.d
    max_iter = args.max_iter
    cce(path, d, max_iter)