from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import remove_classes
from wyrm.processing import swapaxes


class TestRemoveClasses(unittest.TestCase):

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

    def test_remove_classes(self):
        """Removing Classes."""
        # normal case
        dat = remove_classes(self.dat, [0])
        self.assertEqual(dat.data.shape[0], 3)
        np.testing.assert_array_equal(dat.data, self.dat.data[1:])
        # normal every second
        dat = remove_classes(self.dat, [0, 2])
        self.assertEqual(dat.data.shape[0], 2)
        np.testing.assert_array_equal(dat.data, self.dat.data[[1, 3]])
        # the full epo
        dat = remove_classes(self.dat, range(self.dat.data.shape[0]))
        np.testing.assert_array_equal(dat.data.shape[0], 0)

    def test_remove_classes_with_cnt(self):
        """Remove epochs must raise an exception if called with cnt argument."""
        del(self.dat.class_names)
        with self.assertRaises(AssertionError):
            remove_classes(self.dat, [0, 1])

    def test_remove_classes_swapaxes(self):
        """Remove epochs must work with nonstandard classaxis."""
        dat = remove_classes(swapaxes(self.dat, 0, 2), [0, 2], classaxis=2)
        dat = swapaxes(dat, 0, 2)
        dat2 = remove_classes(self.dat, [0, 2])
        self.assertEqual(dat, dat2)

    def test_remove_classes_copy(self):
        """Remove Classes must not modify argument."""
        cpy = self.dat.copy()
        remove_classes(self.dat, [0, 1])
        self.assertEqual(self.dat, cpy)



if __name__ == '__main__':
    unittest.main()
