from collections import defaultdict
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
from Classifier import Classifier
from sklearn import metrics
from sklearn.datasets import load_iris
from mil.data.datasets.loader import load_data
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel

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
        self._clustering_result_kmeans = None
        self._max_iter = max_iter
        self._kmeans_model = None

    @property
    def d(self):
        return self._d
    
    @d.setter
    def d(self, d):
        self._d = d

    @property
    def clustering_result_kmeans(self):
        return self._clustering_result_kmeans
    
    @clustering_result_kmeans.setter
    def clustering_result_kmeans(self, clustering_result_kmeans):
        self._clustering_result_kmeans = clustering_result_kmeans

    @property
    def kmeans_model(self):
        return self._kmeans_model

    @kmeans_model.setter
    def kmeans_model(self, kmeans_model):
        self._kmeans_model = kmeans_model

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter

    def generate_bag_feature_vector_kmeans(self, concept_class: np.ndarray, len_bags: int, X: List): 
        new_bags_rep = np.zeros((len_bags, self.d))

        # # my k means
        # kmeans = KMEANS(self.d, self.max_iter) 
        # kmeans.fit(concept_class)
        # self.clustering_result_kmeans = kmeans.clusters

        # # sklearn k means
        # kmeans = KMeans(n_clusters=self.d, random_state=0).fit(concept_class)
        # self.clustering_result_kmeans = kmeans.labels_

        # KPCA
        kpca = KernelPCA(n_components=None, kernel='precomputed')
        lambda_kpca = 0.5
        kernel_old = lambda_kpca*pow(np.dot(concept_class, concept_class.T), 2) + (1-lambda_kpca)*rbf_kernel(concept_class)
        old_kpca = kpca.fit_transform(kernel_old)
        kmeans = KMeans(n_clusters=self.d, random_state=0).fit(old_kpca)
        self.clustering_result_kmeans = kmeans.labels_

        # use a dict to match the cluster with the instances
        cluster_dict = defaultdict()
        for i in range(0, self.d):
            index = np.where(self.clustering_result_kmeans == i)
            cluster_dict[i] = concept_class[index]
        for bag in range(0, len_bags):
            new_bag_rep = np.zeros(self.d)
            for instance_index in range(0, len(X[bag])):
                instance = np.array(X[bag][instance_index])
                for key, value in enumerate(cluster_dict): # loop through different clusters
                    for temp in cluster_dict[key]: 
                        if np.array_equal(instance, temp):
                            new_bag_rep[key] = 1
            new_bags_rep[bag] = new_bag_rep
        return new_bags_rep

    def fit(self, concept_class: np.ndarray, len_bags: int, X: List, labels: np.ndarray):
        
        new_bags_rep_kmeans = self.generate_bag_feature_vector_kmeans(concept_class, len_bags, X)
        # self.kmeans_model = svm.SVC(kernel="rbf").fit(new_bags_rep_kmeans, labels) 
        kernel = 1.0 * RBF(1.0)
        self.kmeans_model = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(new_bags_rep_kmeans, labels)

    def predict(self, test_concept_class: np.ndarray, len_bags: int, bags_test: List):
        new_bags_rep_kmeans = self.generate_bag_feature_vector_kmeans(test_concept_class, len_bags, bags_test)
        prediction_kmeans = self.kmeans_model.predict(new_bags_rep_kmeans)
        return prediction_kmeans

def load_musk1():
    return load_data('/Users/ningjia/Desktop/440-Final-Project/dataset/musk1.csv')

(bags_train, y_train), (bags_test, y_test) = load_musk1()
list_bag_train = bags_train.copy()
bags_train = util.config_irregular_list(bags_train)
y_train = np.array(y_train)
# bags_test = util.config_irregular_list(bags_test)
y_test = np.array(y_test)
concept_class = util.generate_concept_class(bags_train)
test_concept_class = util.generate_concept_class(bags_test)
cce = CCE(5, 3000) 
#7: 0.6842
cce.fit(concept_class, len(bags_train), list_bag_train, y_train)
y_pred = cce.predict(test_concept_class, len(bags_test), bags_test)
print(y_test)    
print("accuracy:", metrics.accuracy_score(y_test, y_pred))


# if __name__ == '__main__':
#     # Set up argparse arguments
#     parser = argparse.ArgumentParser(description='Run CCE algorithm.')
#     parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
#     parser.add_argument('d', type=int, help='The number of clusters for the clustering algorithm.')
#     args = parser.parse_args()
#     if args.d < 0 :
#        raise argparse.ArgumentTypeError('d must be a positive number.')
#     path = os.path.expanduser(args.path)
#     d = args.d
#     # cce(path, sigma, C)