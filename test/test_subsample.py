import unittest

import numpy as np

from wyrm.misc import Cnt
from wyrm.misc import subsample


class TestSubsample(unittest.TestCase):

    def setUp(self):
        raw = np.arange(2000).reshape(-1, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        fs = 100
        marker = [[100, 'foo'], [200, 'bar']]
        self.cnt = Cnt(raw, fs, channels, marker)

    def test_subsampling(self):
        """Subsampling to 10Hz."""
        cnt = subsample(self.cnt, 10)
        # check if the new fs is correct
        self.assertEqual(cnt.fs, 10.)
        # no channels must have been deleted
        np.testing.assert_array_equal(self.cnt.channels, cnt.channels)
        self.assertEqual(self.cnt.data.shape[-1], cnt.data.shape[-1])
        # markers must have been modified to the new sample positions
        mrk_times_old = [i for i, j in self.cnt.markers]
        mrk_times_new = [i for i, j in cnt.markers]
        self.assertEqual(mrk_times_old, [i * 10 for i in mrk_times_new])
        # no marker must have been deleted
        self.assertEqual(len(self.cnt.markers), len(cnt.markers))
        # check the actual data
        # after subsampling, data should look like:
        # [[0,   1,  2,  3,  4,  5]
        #  [50, 51, 52, 53, 54, 55]
        #  [...]]
        # so the first column of the resampled data should be all
        # multiples of 50.
        zeros = cnt.data[:, 0] % 50
        self.assertFalse(np.any(zeros))

    def test_whole_number_divisor_check(self):
        """Freq must be a whole number divisor of cnt.fs"""
        with self.assertRaises(AssertionError):
            subsample(self.cnt, 33)
        with self.assertRaises(AssertionError):
            subsample(self.cnt, 101)


if __name__ == '__main__':
    unittest.main()
