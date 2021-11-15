import unittest
import numpy as np
import util

class TestMetrics(unittest.TestCase):
    def test_most_likely_cause_estimator(self):
        bag1 = np.matrix([[2,3],[4,5],[7,9]])
        concept1 = np.array([4,5])
        similarity1 = np.exp(0)
        self.assertEqual(similarity1, util.most_likely_cause_estimator(concept1, bag1, 0.5))

        bag2 = np.matrix([[2,3],[4,5],[7,9]])
        concept2 = np.array([5,3])
        similarity2 = 2.0611536224385504e-09
        self.assertEqual(similarity2, util.most_likely_cause_estimator(concept2, bag2, 0.5))
