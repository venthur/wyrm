from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import append_epo
from wyrm.processing import swapaxes


class TestAppendEpo(unittest.TestCase):

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
        self.dat.class_names = ['zero', 'one', 'two']

    def test_append_epo(self):
        """append_epo."""
        dat = append_epo(self.dat, self.dat)
        self.assertEqual(dat.data.shape[0], 2*self.dat.data.shape[0])
        self.assertEqual(len(dat.axes[0]), 2*len(self.dat.axes[0]))
        np.testing.assert_array_equal(dat.data, np.concatenate([self.dat.data, self.dat.data], axis=0))
        np.testing.assert_array_equal(dat.axes[0], np.concatenate([self.dat.axes[0], self.dat.axes[0]]))
        self.assertEqual(dat.class_names, self.dat.class_names)

    def test_append_epo_with_extra(self):
        """append_epo with extra must work with list and ndarrays."""
        self.dat.a = range(10)
        self.dat.b = np.arange(10)
        dat = append_epo(self.dat, self.dat, extra=['a', 'b'])
        self.assertEqual(dat.a, range(10) + range(10))
        np.testing.assert_array_equal(dat.b, np.concatenate([np.arange(10), np.arange(10)]))

    def test_append_epo_with_different_class_names(self):
        """test_append must raise a ValueError if class_names are different."""
        a = self.dat.copy()
        a.class_names = a.class_names[:-1]
        with self.assertRaises(ValueError):
            append_epo(a, self.dat)
            append_epo(self.dat, a)

    def test_append_epo_swapaxes(self):
        """append_epo must work with nonstandard timeaxis."""
        dat = append_epo(swapaxes(self.dat, 0, 2), swapaxes(self.dat, 0, 2), classaxis=2)
        dat = swapaxes(dat, 0, 2)
        dat2 = append_epo(self.dat, self.dat)
        self.assertEqual(dat, dat2)

    def test_append_epo_copy(self):
        """append_epo means must not modify argument."""
        cpy = self.dat.copy()
        append_epo(self.dat, self.dat)
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
