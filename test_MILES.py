import unittest
import numpy as np
from MILES import MILES

class TestMetrics(unittest.TestCase):
    def test_preprocess_data(self):
        bags = np.array([[[2,3],[4,5]], [[3,3],[2,5]], [[4,4],[1,5]]])
        labels = np.array([1,0,1])
        expected_res = np.array([[[2,3],[4,5]], [[4,4],[1,5]], [[3,3],[2,5]]])
        expected_labels = np.array([1,1,0])
        e = MILES(0.5, 0.3)
        res = e.preprocess_data(bags, labels)
        self.assertTrue((res[0] == expected_res).all())
        self.assertTrue((res[1] == expected_labels).all())

    def test_embed_one_bag(self):
        bag1 = np.matrix([[2,3],[4,5]])
        concept1 = np.matrix([[2,3],[4,5],[2,2],[3,4]])
        expected_res = np.array([1.00000000e+00,1.00000000e+00,1.83156389e-02,3.35462628e-04])
        e = MILES(0.5, 0.3)
        res = e.embed_one_bag(concept1, bag1, 0.5)
        for i in range(0, len(res)):
            self.assertAlmostEqual(res[i], expected_res[i])

    def test_embed_all_bags(self):
        bags = np.array([[[2,3],[4,5]], [[3,3],[2,5]], [[4,4],[1,5]]])
        concepts = np.array([[2,3], [4,5], [3,3], [2,5], [4,4], [1,5]])
        expected_res = np.array([[1.00000000e+00, 1.83156389e-02, 2.06115362e-09],
                            [1.00000000e+00, 1.12535175e-07, 1.83156389e-02],
                            [1.83156389e-02, 1.00000000e+00, 3.35462628e-04],
                            [1.12535175e-07, 1.00000000e+00, 1.83156389e-02],
                            [1.83156389e-02, 3.35462628e-04, 1.00000000e+00],
                            [2.06115362e-09, 1.83156389e-02, 1.00000000e+00]])
        e = MILES(0.5, 0.3)
        res = e.embed_all_bags(concepts, bags, 0.5)
        for i in range(0, len(concepts)): # loop through rows
            for j in range(0, len(bags)): # loop through columns
                self.assertAlmostEquals(expected_res[i][j], res[i][j])
                            