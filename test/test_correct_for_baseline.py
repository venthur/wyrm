from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import correct_for_baseline
from wyrm.processing import swapaxes


class TestCorrectForBaseline(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        classes = [0, 0, 0]
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(-1000, 0, 10, endpoint=False)
        # three cnts: 1s, -1s, and 0s
        data = np.array([ones, ones * -1, ones * 0])
        self.dat = Data(data, [classes, time, channels], ['class', 'time', 'channels'], ['#', 'ms', '#'])

    def test_correct_for_baseline_epo(self):
        """Test baselineing w/ epo like."""
        # normal case
        dat = correct_for_baseline(self.dat, [-500, 0])
        np.testing.assert_array_equal(np.zeros((3, 10, 5)), dat.data)
        # the full dat interval
        dat = correct_for_baseline(self.dat, [dat.axes[-2][0], dat.axes[-2][-1]])
        np.testing.assert_array_equal(np.zeros((3, 10, 5)), dat.data)

    def test_correct_for_baseline_cnt(self):
        """Test baselineing w/ cnt like."""
        data = self.dat.data.reshape(30, 5)
        axes = [np.linspace(-1000, 2000, 30, endpoint=False), self.dat.axes[-1]]
        units = self.dat.units[1:]
        names = self.dat.names[1:]
        dat = self.dat.copy(data=data, axes=axes, names=names, units=units)
        dat2 = correct_for_baseline(dat, [-1000, 0])
        np.testing.assert_array_equal(dat2.data, dat.data - 1)

    def test_ival_checks(self):
        """Test for malformed ival parameter."""
        with self.assertRaises(AssertionError):
            correct_for_baseline(self.dat, [0, -1])
        with self.assertRaises(AssertionError):
            correct_for_baseline(self.dat, [self.dat.axes[-2][0]-1, 0])
        with self.assertRaises(AssertionError):
            correct_for_baseline(self.dat, [0, self.dat.axes[-2][1]+1])

    def test_correct_for_baseline_copy(self):
        """Correct for baseline must not modify dat argument."""
        cpy = self.dat.copy()
        correct_for_baseline(self.dat, [-1000, 0])
        self.assertEqual(cpy, self.dat)

    def test_correct_for_baseline_swapaxes(self):
        """Correct for baseline must work with nonstandard timeaxis."""
        dat = correct_for_baseline(swapaxes(self.dat, 0, 1), [-1000, 0], timeaxis=0)
        dat = swapaxes(dat, 0, 1)
        dat2 = correct_for_baseline(self.dat, [-1000, 0])
        self.assertEqual(dat, dat2)


if __name__ == '__main__':
    unittest.main()
