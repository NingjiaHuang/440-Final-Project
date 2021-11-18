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

    def test_generate_concept_class(self):
        bags = np.array([[[ 4.87496681,-4.4644499 ],
                            [-4.8542007,5.31296979]],
                            [[ 4.97653599, 5.40067439],
                            [ 3.77876696,5.7696428 ]],
                            [[-6.89473217,3.9623194 ],
                            [-5.35726469,-3.30651547]],
                            [1000000, 1000000]])
        expected_res = np.array([[ 4.87496681, -4.4644499 ],
                            [-4.8542007, 5.31296979],
                            [ 4.97653599, 5.40067439],
                            [ 3.77876696, 5.7696428 ],
                            [-6.89473217, 3.9623194 ],
                            [-5.35726469, -3.30651547]])
        res = util.generate_concept_class(bags)
        self.assertTrue((res == expected_res).all())
        array = np.array([])
        self.assertEqual(type(res), type(array))
        self.assertEqual(type(res[1]), type(array))