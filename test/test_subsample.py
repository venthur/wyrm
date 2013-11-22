from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import subsample
from wyrm.processing import swapaxes

class TestSubsample(unittest.TestCase):

    def setUp(self):
        raw = np.arange(2000).reshape(-1, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(0, 4000, 400, endpoint=False)
        fs = 100
        marker = [[100, 'foo'], [200, 'bar']]
        self.dat = Data(raw, [time, channels], ['time', 'channels'], ['ms', '#'])
        self.dat.fs = fs
        self.dat.markers = marker

    def test_subsampling(self):
        """Subsampling to 10Hz."""
        dat = subsample(self.dat, 10)
        # check if the new fs is correct
        self.assertEqual(dat.fs, 10.)
        # check if data and axes have the same length
        self.assertEqual(len(dat.axes[-1]), dat.data.shape[-1])
        # no channels must have been deleted
        np.testing.assert_array_equal(self.dat.axes[-1], dat.axes[-1])
        self.assertEqual(self.dat.data.shape[-1], dat.data.shape[-1])
        # markers must not have been modified
        self.assertEqual(self.dat.markers, dat.markers)
        # no marker must have been deleted
        self.assertEqual(len(self.dat.markers), len(dat.markers))
        # check the actual data
        # after subsampling, data should look like:
        # [[0,   1,  2,  3,  4,  5]
        #  [50, 51, 52, 53, 54, 55]
        #  [...]]
        # so the first column of the resampled data should be all
        # multiples of 50.
        zeros = dat.data[:, 0] % 50
        self.assertFalse(np.any(zeros))

    def test_subsample_with_epo(self):
        """subsample must work with epoched data."""
        data = np.array([self.dat.data, self.dat.data, self.dat.data])
        axes = [np.arange(3), self.dat.axes[0], self.dat.axes[1]]
        names = ['class', 'time', 'channel']
        units = ['#', 'ms', '#']
        dat = self.dat.copy(data=data, axes=axes, names=names, units=units)
        dat = subsample(dat, 10)
        self.assertEqual(dat.fs, 10)
        self.assertEqual(dat.data.ndim, 3)
        self.assertEqual(len(dat.axes[1]), dat.data.shape[1])
        self.assertEqual(dat.data.shape[1], self.dat.data.shape[0] / 10)

    def test_whole_number_divisor_check(self):
        """Freq must be a whole number divisor of dat.fs"""
        with self.assertRaises(AssertionError):
            subsample(self.dat, 33)
        with self.assertRaises(AssertionError):
            subsample(self.dat, 101)

    def test_has_fs_check(self):
        """subsample must raise an exception if .fs attribute is not found."""
        with self.assertRaises(AssertionError):
            del(self.dat.fs)
            subsample(self.dat, 10)

    def test_axes_and_data_have_same_len_check(self):
        """subsample must raise an error if the timeaxis and data have not the same lengh."""
        with self.assertRaises(AssertionError):
            self.dat.axes[-2] = self.dat.axes[-2][1:]
            subsample(self.dat, 10)

    def test_subsample_copy(self):
        """Subsample must not modify argument."""
        cpy = self.dat.copy()
        subsample(self.dat, 10)
        self.assertEqual(cpy, self.dat)

    def test_subsample_swapaxes(self):
        """subsample must work with nonstandard timeaxis."""
        dat = subsample(swapaxes(self.dat, 0, 1), 10, timeaxis=1)
        dat = swapaxes(dat, 0, 1)
        dat2 = subsample(self.dat, 10)
        self.assertEqual(dat, dat2)


if __name__ == '__main__':
    unittest.main()
