import unittest

import numpy as np

from wyrm.misc import Cnt
from wyrm.misc import subsample


class TestSubsample(unittest.TestCase):

    def setUp(self):
        raw = np.arange(2000).reshape(-1, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        fs = 10
        marker = [[100, 'foo'], [200, 'bar']]
        self.cnt = Cnt(raw, fs, channels, marker)

    def test_subsampling_by_10(self):
        """Subsampling by factor 10."""
        cnt = subsample(self.cnt, 10)
        # check if the new fs is correct
        self.assertEqual(cnt.fs, self.cnt.fs / 10)
        # no channels must have been deleted
        np.testing.assert_array_equal(self.cnt.channels, cnt.channels)
        self.assertEqual(self.cnt.data.shape[-1], cnt.data.shape[-1])
        # markers must have been modified to the new sample positions
        mrk_times_old = [i for i, j in self.cnt.markers]
        mrk_times_new = [i for i, j in cnt.markers]
        self.assertEqual(mrk_times_old, [i * 10 for i in mrk_times_new])
        # check the actual data
        # after subsampling, data should look like:
        # [[0,   1,  2,  3,  4,  5]
        #  [50, 51, 52, 53, 54, 55]
        #  [...]]
        # so the first column of the resampled data should be all
        # multiples of 50.
        zeros = cnt.data[:, 0] % 50
        self.assertFalse(np.any(zeros))

if __name__ == '__main__':
    unittest.main()
