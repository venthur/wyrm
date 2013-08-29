from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import jumping_means
from wyrm.processing import swapaxes


class TestJumpingMeans(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        # cnt with 1, 2, 3
        cnt = np.append(ones, ones*2, axis=0)
        cnt = np.append(cnt, ones*3, axis=0)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(0, 3000, 30, endpoint=False)
        classes = [0, 1, 2, 1]
        # four cnts: 1s, -1s, and 0s
        data = np.array([cnt * 0, cnt * 1, cnt * 2, cnt * 0])
        self.dat = Data(data, [classes, time, channels], ['class', 'time', 'channel'], ['#', 'ms', '#'])

    def test_jumping_means(self):
        """Jumping means."""
        # with several ivals
        dat = jumping_means(self.dat, [[0, 1000], [1000, 2000], [2000, 3000]])
        newshape = list(self.dat.data.shape)
        newshape[1] = 3
        self.assertEqual(list(dat.data.shape), newshape)
        # first epo (0)
        self.assertEqual(dat.data[0, 0, 0], 0)
        self.assertEqual(dat.data[0, 1, 0], 0)
        self.assertEqual(dat.data[0, 2, 0], 0)
        # second epo (1)
        self.assertEqual(dat.data[1, 0, 0], 1)
        self.assertEqual(dat.data[1, 1, 0], 2)
        self.assertEqual(dat.data[1, 2, 0], 3)
        # third epo (2)
        self.assertEqual(dat.data[2, 0, 0], 2)
        self.assertEqual(dat.data[2, 1, 0], 4)
        self.assertEqual(dat.data[2, 2, 0], 6)
        # fourth epo (0)
        self.assertEqual(dat.data[3, 0, 0], 0)
        self.assertEqual(dat.data[3, 1, 0], 0)
        self.assertEqual(dat.data[3, 2, 0], 0)
        # with one ival
        dat = jumping_means(self.dat, [[0, 1000]])
        newshape = list(self.dat.data.shape)
        newshape[1] = 1
        self.assertEqual(list(dat.data.shape), newshape)
        # first epo (0)
        self.assertEqual(dat.data[0, 0, 0], 0)
        # second epo (1)
        self.assertEqual(dat.data[1, 0, 0], 1)
        # third epo (2)
        self.assertEqual(dat.data[2, 0, 0], 2)
        # fourth epo (0)
        self.assertEqual(dat.data[3, 0, 0], 0)

    def test_jumping_means_with_cnt(self):
        """jumping_means must work with cnt argument."""
        data = self.dat.data[1]
        axes = self.dat.axes[1:]
        names = self.dat.names[1:]
        units = self.dat.units[1:]
        dat = self.dat.copy(data=data, axes=axes, names=names, units=units)
        dat = jumping_means(dat, [[0, 1000], [1000, 2000]])
        self.assertEqual(dat.data[0, 0], 1)
        self.assertEqual(dat.data[1, 0], 2)

    def test_jumping_means_swapaxes(self):
        """jumping means must work with nonstandard timeaxis."""
        dat = jumping_means(swapaxes(self.dat, 1, 2), [[0, 1000]], timeaxis=2)
        dat = swapaxes(dat, 1, 2)
        dat2 = jumping_means(self.dat, [[0, 1000]])
        self.assertEqual(dat, dat2)

    def test_jumping_means_copy(self):
        """jumping means must not modify argument."""
        cpy = self.dat.copy()
        jumping_means(self.dat, [[0, 1000]])
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
