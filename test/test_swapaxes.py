from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import swapaxes

class TestSwapaxes(unittest.TestCase):

    def setUp(self):
        raw = np.arange(2000).reshape(-1, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(0, 4000, 400, endpoint=False)
        fs = 100
        marker = [[100, 'foo'], [200, 'bar']]
        self.dat = Data(raw, [time, channels], ['time', 'channels'], ['ms', '#'])
        self.dat.fs = fs
        self.dat.markers = marker

    def test_swapaxes(self):
        """Swapping axes."""
        new = swapaxes(self.dat, 0, 1)
        self.assertTrue((new.axes[0] == self.dat.axes[1]).all())
        self.assertTrue((new.axes[1] == self.dat.axes[0]).all())
        self.assertEqual(new.names[0], self.dat.names[1])
        self.assertEqual(new.names[1], self.dat.names[0])
        self.assertEqual(new.units[0], self.dat.units[1])
        self.assertEqual(new.units[1], self.dat.units[0])
        self.assertEqual(new.data.shape[::-1], self.dat.data.shape)
        np.testing.assert_array_equal(new.data.swapaxes(0, 1), self.dat.data)

    def test_swapaxes_copy(self):
        """Swapaxes must not modify argument."""
        cpy = self.dat.copy()
        swapaxes(self.dat, 0, 1)
        self.assertEqual(cpy, self.dat)

    def test_swapaxes_twice(self):
        """Swapping the same axes twice must result in original."""
        dat = swapaxes(self.dat, 0, 1)
        dat = swapaxes(dat, 0, 1)
        self.assertEqual(dat, self.dat)


if __name__ == '__main__':
    unittest.main()
