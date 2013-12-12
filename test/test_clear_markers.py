from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import clear_markers
from wyrm.processing import swapaxes


class TestClearMarkers(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        # cnt with 1, 2, 3
        cnt = np.append(ones, ones*2, axis=0)
        cnt = np.append(cnt, ones*3, axis=0)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(-1000, 2000, 30, endpoint=False)
        self.good_markers = [[-1000, 'a'], [-999, 'b'], [0, 'c'], [1999.9999999, 'd']]
        bad_markers = [[-1001, 'x'], [2000, 'x']]
        markers = self.good_markers[:]
        markers.extend(bad_markers)
        classes = [0, 1, 2, 1]
        # four cnts: 1s, -1s, and 0s
        data = np.array([cnt * 0, cnt * 1, cnt * 2, cnt * 0])
        self.dat = Data(data, [classes, time, channels], ['class', 'time', 'channel'], ['#', 'ms', '#'])
        self.dat.markers = markers
        self.dat.fs = 10

    def test_clear_markers(self):
        """Clear markers."""
        dat = clear_markers(self.dat)
        self.assertEqual(dat.markers, self.good_markers)

    def test_clear_emtpy_markers(self):
        """Clearing emtpy markers has no effect."""
        dat = self.dat.copy()
        dat.markers = []
        dat2 = clear_markers(dat)
        self.assertEqual(dat, dat2)

    def test_clear_nonexisting_markers(self):
        """Clearing emtpy markers has no effect."""
        dat = self.dat.copy()
        del dat.markers
        dat2 = clear_markers(dat)
        self.assertEqual(dat, dat2)

    def test_clear_markers_w_empty_data(self):
        """Clearing emtpy dat should remove all markers."""
        dat = self.dat.copy()
        dat.data = np.array([])
        dat2 = clear_markers(dat)
        self.assertEqual(dat2.markers, [])

    def test_clear_markes_swapaxes(self):
        """clear_markers must work with nonstandard timeaxis."""
        dat = clear_markers(swapaxes(self.dat, 1, 2), timeaxis=2)
        dat = swapaxes(dat, 1, 2)
        dat2 = clear_markers(self.dat)
        self.assertEqual(dat, dat2)

    def test_clear_markers_copy(self):
        """clear_markers must not modify argument."""
        cpy = self.dat.copy()
        clear_markers(self.dat)
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
