from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import select_epochs
from wyrm.processing import swapaxes


class TestSelectEpochs(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.linspace(0, 1000, 10, endpoint=False)
        classes = [0, 1, 2, 1]
        class_names = ['zeros', 'ones', 'twoes']
        # four cnts: 0s, 1s, -1s, and 0s
        data = np.array([ones * 0, ones * 1, ones * 2, ones * 0])
        self.dat = Data(data, [classes, time, channels], ['class', 'time', 'channel'], ['#', 'ms', '#'])
        self.dat.class_names = class_names

    def test_select_epochs(self):
        """Selecting Epochs."""
        # normal case
        dat = select_epochs(self.dat, [0])
        self.assertEqual(dat.data.shape[0], 1)
        np.testing.assert_array_equal(dat.data, self.dat.data[[0]])
        np.testing.assert_array_equal(dat.axes[0], self.dat.axes[0][0])
        # normal every second
        dat = select_epochs(self.dat, [0, 2])
        self.assertEqual(dat.data.shape[0], 2)
        np.testing.assert_array_equal(dat.data, self.dat.data[::2])
        np.testing.assert_array_equal(dat.axes[0], self.dat.axes[0][::2])
        # the full epo
        dat = select_epochs(self.dat, range(self.dat.data.shape[0]))
        np.testing.assert_array_equal(dat.data, self.dat.data)
        np.testing.assert_array_equal(dat.axes[0], self.dat.axes[0])
        # remove one
        dat = select_epochs(self.dat, [0], invert=True)
        self.assertEqual(dat.data.shape[0], 3)
        np.testing.assert_array_equal(dat.data, self.dat.data[1:])
        np.testing.assert_array_equal(dat.axes[0], self.dat.axes[0][1:])
        # remove every second
        dat = select_epochs(self.dat, [0, 2], invert=True)
        self.assertEqual(dat.data.shape[0], 2)
        np.testing.assert_array_equal(dat.data, self.dat.data[1::2])
        np.testing.assert_array_equal(dat.axes[0], self.dat.axes[0][1::2])

    def test_select_epochs_with_cnt(self):
        """Select epochs must raise an exception if called with cnt argument."""
        del(self.dat.class_names)
        with self.assertRaises(AssertionError):
            select_epochs(self.dat, [0, 1])

    def test_select_epochs_swapaxes(self):
        """Select epochs must work with nonstandard classaxis."""
        dat = select_epochs(swapaxes(self.dat, 0, 2), [0, 1], classaxis=2)
        dat = swapaxes(dat, 0, 2)
        dat2 = select_epochs(self.dat, [0, 1])
        self.assertEqual(dat, dat2)

    def test_select_epochs_copy(self):
        """Select Epochs must not modify argument."""
        cpy = self.dat.copy()
        select_epochs(self.dat, [0, 1])
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
