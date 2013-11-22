from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import append_cnt
from wyrm.processing import swapaxes


class TestAppendCnt(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        # cnt with 1, 2, 3
        cnt = np.append(ones, ones*2, axis=0)
        cnt = np.append(cnt, ones*3, axis=0)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(0, 3000, 30, endpoint=False)
        self.dat = Data(cnt, [time, channels], ['time', 'channel'], ['ms', '#'])
        self.dat.markers = [[0, 'a'], [1, 'b']]
        self.dat.fs = 10

    def test_append_cnt(self):
        """append_cnt."""
        dat = append_cnt(self.dat, self.dat)
        self.assertEqual(dat.data.shape[0], 2*self.dat.data.shape[0])
        self.assertEqual(len(dat.axes[0]), 2*len(self.dat.axes[0]))
        np.testing.assert_array_equal(dat.data, np.concatenate([self.dat.data, self.dat.data], axis=0))
        np.testing.assert_array_equal(dat.axes[0], np.linspace(0, 6000, 60, endpoint=False))
        self.assertEqual(dat.markers, self.dat.markers + map(lambda x: [x[0] + 3000, x[1]], self.dat.markers))

    def test_append_cnt_with_extra(self):
        """append_cnt with extra must work with list and ndarrays."""
        self.dat.a = range(10)
        self.dat.b = np.arange(10)
        dat = append_cnt(self.dat, self.dat, extra=['a', 'b'])
        self.assertEqual(dat.a, range(10) + range(10))
        np.testing.assert_array_equal(dat.b, np.concatenate([np.arange(10), np.arange(10)]))
        self.assertEqual(dat.markers, self.dat.markers + map(lambda x: [x[0] + 3000, x[1]], self.dat.markers))

    def test_append_cnt_swapaxes(self):
        """append_cnt must work with nonstandard timeaxis."""
        dat = append_cnt(swapaxes(self.dat, 0, 1), swapaxes(self.dat, 0, 1), timeaxis=1)
        dat = swapaxes(dat, 0, 1)
        dat2 = append_cnt(self.dat, self.dat)
        self.assertEqual(dat, dat2)

    def test_append_cnt_copy(self):
        """append_cnt means must not modify argument."""
        cpy = self.dat.copy()
        append_cnt(self.dat, self.dat)
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
