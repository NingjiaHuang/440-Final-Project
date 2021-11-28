import unittest
import numpy as np
from CCE import CCE

class TestMetrics(unittest.TestCase):
    def test_overlap(self):
        bag = np.array([[1,2],[3,4],[5,6]])
        group_k = np.array([[1,2],[4,4],[5,7],[8,9]])
        cce = CCE(5, 1000)
        self.assertEqual(1, cce.overlap(bag, group_k))
        bag_2 = np.array([[1,2],[3,4],[5,6]])
        group_k_2 = np.array([[1,3],[4,4],[5,7],[8,9]])
        self.assertEqual(0, cce.overlap(bag_2, group_k_2))
        bag_3 = np.array([])
        group_k_3 = np.array([[1,3],[4,4],[5,7],[8,9]])
        self.assertEqual(0, cce.overlap(bag_3, group_k_3))
    
    def test_select_by_index(self):
        cce = CCE(5, 1000)
        cluster_index = [0, 1, 0, 0, 1]
        concept_class = np.array([[1,2],[2,2],[3,1],[5,4],[2,3]])
        expected_output = [[[1,2],[3,1],[5,4]],[[2,2],[2,3]]]
        output = cce.select_by_index(cluster_index, 2, concept_class)
        for i in range(0, len(output)): 
            for j in range(0, len(output[i])): 
                for k in range(0, len(output[i][j])):
                    self.assertEqual(expected_output[i][j][k], output[i][j][k])

    def test_majority_vote(self):
        cce = CCE(5, 1000)
        list_of_prediction = [[1,1,0],[0,1,0],[0,0,1]]
        expected_output = np.array([0,1,0])
        self.assertTrue(np.array_equal(expected_output, cce.majority_vote(list_of_prediction)))