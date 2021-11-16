import unittest
import numpy as np
from MILES import MILES

class TestMetrics(unittest.TestCase):
    def test_embed_one_bag(self):
        bag1 = np.matrix([[2,3],[4,5]])
        concept1 = np.matrix([[2,3],[4,5],[2,2],[3,4]])
        expected_res = np.array([1.00000000e+00,1.00000000e+00,1.83156389e-02,3.35462628e-04])
        e = MILES()
        res = e.embed_one_bag(concept1, bag1, 0.5)
        for i in range(0, len(res)):
            self.assertAlmostEqual(res[i], expected_res[i])
