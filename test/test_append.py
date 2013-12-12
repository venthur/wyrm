from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import append
from wyrm.processing import swapaxes


class TestAppend(unittest.TestCase):

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

    def test_append(self):
        """Append."""
        dat = append(self.dat, self.dat)
        self.assertEqual(dat.data.shape[0], 2*self.dat.data.shape[0])
        self.assertEqual(len(dat.axes[0]), 2*len(self.dat.axes[0]))
        np.testing.assert_array_equal(dat.data, np.concatenate([self.dat.data, self.dat.data], axis=0))
        np.testing.assert_array_equal(dat.axes[0], np.concatenate([self.dat.axes[0], self.dat.axes[0]]))

    def test_append_with_extra(self):
        """append with extra must work with list and ndarrays."""
        self.dat.a = range(10)
        self.dat.b = np.arange(10)
        dat = append(self.dat, self.dat, extra=['a', 'b'])
        self.assertEqual(dat.a, range(10) + range(10))
        np.testing.assert_array_equal(dat.b, np.concatenate([np.arange(10), np.arange(10)]))

    def test_append_with_different_extra_types(self):
        """append must throw a TypeError if extra-types don't match."""
        a = self.dat.copy()
        b = self.dat.copy()
        a.a = range(10)
        b.a = np.arange(10)
        with self.assertRaises(TypeError):
            append(a, b, extra=['a'])

    def test_append_with_wrong_known_attributes(self):
        # .data dimensions must match
        a = self.dat.copy()
        a.data = a.data[np.newaxis, ...]
        with self.assertRaises(AssertionError):
            append(self.dat, a)
        # .data.shape must match for all axes except the ones beeing
        # appended
        a = self.dat.copy()
        a.data = a.data[..., :-1]
        with self.assertRaises(AssertionError):
            append(self.dat, a)
        # .axes must be equal for all axes except the ones beeing
        # appended
        a = self.dat.copy()
        a.axes[-1][0] = 'foo'
        with self.assertRaises(AssertionError):
            append(self.dat, a)
        # names must be equal
        a = self.dat.copy()
        a.names[0] = 'foo'
        with self.assertRaises(AssertionError):
            append(self.dat, a)
        # units must be equal
        a = self.dat.copy()
        a.units[0] = 'foo'
        with self.assertRaises(AssertionError):
            append(self.dat, a)

    def test_append_with_unsupported_extra_types(self):
        """append must trhow a TypeError if extra-type is unsupported."""
        self.dat.a = {'foo' : 'bar'}
        with self.assertRaises(TypeError):
            append(self.dat, self.dat, extra=['a'])

    def test_append_with_cnt(self):
        """append must work with cnt argument."""
        data = self.dat.data[1]
        axes = self.dat.axes[1:]
        names = self.dat.names[1:]
        units = self.dat.units[1:]
        dat = self.dat.copy(data=data, axes=axes, names=names, units=units)
        dat2 = append(dat, dat)
        self.assertEqual(dat2.data.shape[0], 2*dat.data.shape[0])
        self.assertEqual(len(dat2.axes[0]), 2*len(dat.axes[0]))
        np.testing.assert_array_equal(dat2.data, np.concatenate([dat.data, dat.data], axis=0))
        np.testing.assert_array_equal(dat2.axes[0], np.concatenate([dat.axes[0], dat.axes[0]], axis=0))

    def test_append_with_negative_axis(self):
        """Append must work correctly with a negative axis."""
        dat2 = self.dat.copy()
        dat2.data = dat2.data[:-1, ...]
        dat2.axes[0] = dat2.axes[2][:-1]
        a = append(self.dat, dat2, axis=0)
        b = append(self.dat, dat2, axis=-3)
        self.assertEqual(a, b)

    def test_append_swapaxes(self):
        """append must work with nonstandard timeaxis."""
        dat = append(swapaxes(self.dat, 0, 2), swapaxes(self.dat, 0, 2), axis=2)
        dat = swapaxes(dat, 0, 2)
        dat2 = append(self.dat, self.dat)
        self.assertEqual(dat, dat2)

    def test_append_copy(self):
        """append means must not modify argument."""
        cpy = self.dat.copy()
        append(self.dat, self.dat)
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
