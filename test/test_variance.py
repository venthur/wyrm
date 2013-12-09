from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import variance
from wyrm.processing import swapaxes


class TestVariance(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        # epo with 0, 1, 2
        data = np.array([0*ones, ones, 2*ones])
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(0, 1000, 10, endpoint=False)
        classes = [0, 1, 2]
        self.dat = Data(data, [classes, time, channels], ['class', 'time', 'channel'], ['#', 'ms', '#'])

    def test_variance(self):
        """Variance."""
        dat = variance(self.dat)
        # test the resulting dat has one axis less (the middle one)
        self.assertEqual(dat.data.shape, self.dat.data.shape[::2])
        # each epoch should have a variance of zero, test if the var of
        # all epochs is 0
        self.assertEqual(dat.data.var(), 0)
        self.assertEqual(len(dat.axes), len(self.dat.axes)-1)

    def test_variance_with_cnt(self):
        """variance must work with cnt argument."""
        data = self.dat.data[1]
        axes = self.dat.axes[1:]
        names = self.dat.names[1:]
        units = self.dat.units[1:]
        dat = self.dat.copy(data=data, axes=axes, names=names, units=units)
        dat = variance(dat)
        self.assertEqual(dat.data.var(), 0)
        self.assertEqual(len(dat.axes), len(self.dat.axes)-2)

    def test_variance_swapaxes(self):
        """variance must work with nonstandard timeaxis."""
        dat = variance(swapaxes(self.dat, 1, 2), timeaxis=2)
        # we don't swap back here as variance removes the timeaxis
        dat2 = variance(self.dat)
        self.assertEqual(dat, dat2)

    def test_variance_copy(self):
        """variance must not modify argument."""
        cpy = self.dat.copy()
        variance(self.dat)
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
