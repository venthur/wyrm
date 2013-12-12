#!/usr/bin/env python


import unittest

import numpy as np

from wyrm.types import Data


class TestData(unittest.TestCase):

    def setUp(self):
        self.data = np.arange(20).reshape(4, 5)
        self.axes = [np.arange(4), np.arange(5)]
        self.names = ['foo', 'bar']
        self.units = ['u1', 'u2']

    def test_init(self):
        """Test init with correct values."""
        d = Data(self.data, self.axes, self.names, self.units)
        np.testing.assert_array_equal(d.data, self.data)
        for a, b in zip(d.axes, self.axes):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(self.names, d.names)
        self.assertEqual(self.units, d.units)

    def test_init_emtpy(self):
        """Test init with empty data."""
        d = Data(np.array([]), [], [], [])
        self.assertFalse(d)

    def test_init_emtpy2(self):
        """With emtpy data it does not matter what the rest is."""
        d = Data(np.array([]), [1, 2, 3], [1, 1, 2], [1, 2, 3])
        self.assertFalse(d)

    def test_init_with_inconsistent_values(self):
        """Test init with inconsistent values."""
        data = self.data[np.newaxis, :]
        with self.assertRaises(AssertionError):
            Data(data, self.axes, self.names, self.units)
        axes = self.axes[:]
        axes[0] = np.arange(100)
        with self.assertRaises(AssertionError):
            Data(self.data, axes, self.names, self.units)
        names = self.names[:]
        names.append('baz')
        with self.assertRaises(AssertionError):
            Data(self.data, self.axes, names, self.units)
        units = self.units[:]
        units.append('u3')
        with self.assertRaises(AssertionError):
            Data(self.data, self.axes, self.names, units)

    def test_truth_value(self):
        """Test __nonzero__."""
        d = Data(np.array([]), [], [], [])
        self.assertFalse(d)
        d = Data(self.data, self.axes, self.names, self.units)
        self.assertTrue(d)

    def test_equality(self):
        """Test the various (in)equalities."""
        d1 = Data(self.data, self.axes, self.names, self.units)
        # known extra attributes
        d1.markers = [[123, 'foo'], [234, 'bar']]
        d1.fs = 100
        # unknown extra attribute
        d1.foo = 'bar'
        # so far, so equal
        d2 = d1.copy()
        self.assertEqual(d1, d2)
        # different shape
        d2 = d1.copy()
        d2.data = np.arange(20).reshape(5, 4)
        self.assertNotEqual(d1, d2)
        # different data
        d2 = d1.copy()
        d2.data[0, 0] = 42
        self.assertNotEqual(d1, d2)
        # different axes
        d2 = d1.copy()
        d2.axes[0] = np.arange(100)
        self.assertNotEqual(d1, d2)
        # different names
        d2 = d1.copy()
        d2.names[0] = 'baz'
        self.assertNotEqual(d1, d2)
        # different untis
        d2 = d1.copy()
        d2.units[0] = 'u3'
        self.assertNotEqual(d1, d2)
        # different known extra attribute
        d2 = d1.copy()
        d2.markers[0] = [123, 'baz']
        self.assertNotEqual(d1, d2)
        # different known extra attribute
        d2 = d1.copy()
        d2.fs = 10
        self.assertNotEqual(d1, d2)
        # different unknown extra attribute
        d2 = d1.copy()
        d2.baz = 'baz'
        self.assertNotEqual(d1, d2)
        # different new unknown extra attribute
        d2 = d1.copy()
        d2.bar = 42
        self.assertNotEqual(d1, d2)

    def test_eq_and_ne(self):
        """Check if __ne__ is properly implemented."""
        d1 = Data(self.data, self.axes, self.names, self.units)
        d2 = d1.copy()
        # if __eq__ is implemented and __ne__ is not, this evaluates to
        # True!
        self.assertFalse(d1 == d2 and d1 != d2)

    def test_copy(self):
        """Copy must work."""
        d1 = Data(self.data, self.axes, self.names, self.units)
        d2 = d1.copy()
        self.assertEqual(d1, d2)
        # we can't really check of all references to be different in
        # depth recursively, so we only check on the first level
        for k in d1.__dict__:
            self.assertNotEqual(id(getattr(d1, k)), id(getattr(d2, k)))
        d2 = d1.copy(foo='bar')
        self.assertEqual(d2.foo, 'bar')



if __name__ == '__main__':
    unittest.main()

