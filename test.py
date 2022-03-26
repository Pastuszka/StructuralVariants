import main
import unittest
import numpy as np


class TestMeanShift(unittest.TestCase):

    def test_mean_shift_gradient(self):
        rd = np.array([1, 2, 2, 4, 1])
        hb = 1
        hr = np.array([2, 2, 2, 2, 2])
        correct = np.array([0.53526143, 0.07126923, -0.23865122, -0.17096777, -0.19691168])
        result = main.mean_shift_gradient(rd, hb, hr, 2)
        self.assertTrue(np.allclose(correct, result))


if __name__ == '__main__':
    unittest.main()
