import main
import unittest
import numpy as np


class TestMeanShift(unittest.TestCase):

    def test_mean_shift_gradient(self):
        rd = np.array([1, 2, 2, 4, 1])
        hb = 1
        hr = np.array([2, 2, 2, 2, 2])
        correct = np.array([0.53526143, 0.07126923, -0.23865122, -0.17096777, -0.19691168])
        result = main.mean_shift_gradient(rd, hb, hr)
        self.assertTrue(np.allclose(correct, result))

    def test_mean_shift_partition(self):
        rd = np.array([0.5, 0.4, 1, 1, 1, 0])
        gradient = np.array([-1, -1, 1, 0, -1, 1])
        starts, rd_new = main.mean_shift_partition(rd, gradient)
        starts_true = np.array([0, 2, 5])
        new_true = np.array([0.45, 0.45, 1, 1, 1, 0])
        a = np.all(starts_true == starts)
        b = np.all(new_true == rd_new)
        self.assertTrue(a and b)

    def test_reinsert_frozen_segments(self):
        is_frozen = np.array([False, False, False, True, True, True, False, False])
        segment_starts = np.array([0, 1, 3])
        frozen_segment_starts = np.array([3, 5])
        correct_starts = np.array([0, 1, 3, 5, 6])
        all_starts = main.reinsert_frozen_segments(segment_starts, frozen_segment_starts, is_frozen)
        self.assertTrue(np.all(all_starts == correct_starts))

    def test_merge_signals(self):
        rd_corrected = np.array([12, 12, 11, 4, 4, 5, 5, 11])
        segments = [0, 3, 5, 7]
        new_segments = main.merge_signals(rd_corrected, segments)
        self.assertTrue(np.all(np.array([0, 3, 7]) == new_segments))


class TestCalling(unittest.TestCase):

    def test_call_variants(self):
        segments = [0, 2, 4, 6]
        rd = np.array([5, 5, 7, 7, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        called, is_deletion = main.call_variants(segments, rd)
        calls_correct = called[0] == (2,3) and called[1] == (4,5) and len(called) == 2
        is_deletion_correct = np.all(np.array([False, True]) == is_deletion)
        self.assertTrue(calls_correct and is_deletion_correct)

    def test_merge_calls(self):
        rd = [5, 5, 7, 7, 6, 7, 7, 5, 5, 5, 5, 5]
        calls = [(2, 3), (5, 6)]
        is_deletion = [False, False]
        new_calls, new_is_deletion = main.merge_calls(calls, rd, is_deletion)
        calls_correct = new_calls[0] == (2, 6) and len(new_calls) == 1
        is_deletion_correct = new_is_deletion[0] is False and len(new_is_deletion) == 1
        self.assertTrue(calls_correct and is_deletion_correct)


if __name__ == '__main__':
    unittest.main()
