from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import select_ival
from wyrm.processing import swapaxes


class TestSelectIval(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(-1000, 0, 10, endpoint=False)
        classes = [0, 0, 0]
        # three cnts: 1s, -1s, and 0s
        data = np.array([ones, ones * -1, ones * 0])
        self.dat = Data(data, [classes, time, channels], ['class', 'time', 'channel'], ['#', 'ms', '#'])
        self.dat.fs = 10

    def test_select_ival(self):
        """Selecting Intervals."""
        # normal case
        dat = select_ival(self.dat, [-500, 0])
        self.assertEqual(dat.axes[1][0], -500)
        self.assertEqual(dat.axes[1][-1],-100)
        # the full dat interval
        dat = select_ival(self.dat, [self.dat.axes[1][0], self.dat.axes[1][-1] + 1])
        self.assertEqual(dat.axes[1][0], self.dat.axes[1][0])
        self.assertEqual(dat.axes[1][-1], self.dat.axes[1][-1])
        np.testing.assert_array_equal(dat.data, self.dat.data)

    def test_select_ival_with_markers(self):
        """Selecting Intervals with markers."""
        # normal case
        good_markers = [[-499,99, 'x'], [-500, 'x'], [-0.0001, 'x']]
        bad_markers = [[501, 'y'], [0, 'y'], [1, 'y']]
        self.dat.markers = good_markers[:]
        self.dat.markers.extend(bad_markers)
        dat = select_ival(self.dat, [-500, 0])
        self.assertEqual(dat.markers, good_markers)

    def test_ival_checks(self):
        """Test for malformed ival parameter."""
        with self.assertRaises(AssertionError):
            select_ival(self.dat, [0, -1])
        with self.assertRaises(AssertionError):
            select_ival(self.dat, [self.dat.axes[1][0]-1, 0])
        with self.assertRaises(AssertionError):
            select_ival(self.dat, [0, self.dat.axes[1][-1]+1])

    def test_select_ival_copy(self):
        """Select_ival must not modify the argument."""
        cpy = self.dat.copy()
        select_ival(cpy, [-500, 0])
        self.assertEqual(cpy, self.dat)

    def test_select_ival_swapaxes(self):
        """select_ival must work with nonstandard timeaxis."""
        dat = select_ival(swapaxes(self.dat, 0, 1), [-500, 0], timeaxis=0)
        dat = swapaxes(dat, 0, 1)
        dat2 = select_ival(self.dat, [-500, 0])
        self.assertEqual(dat, dat2)

if __name__ == '__main__':
    unittest.main()
