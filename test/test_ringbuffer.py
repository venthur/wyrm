#!/usr/bin/env python


import unittest

import numpy as np

from wyrm.types import RingBuffer


class TestRingBuffer(unittest.TestCase):

    def setUp(self):
        self.rb = RingBuffer(10)

    def test_add_empty(self):
        """Adding an emtpy array must not modify the ringbuffer."""
        # start with emtpy rb
        data0 = self.rb.get()
        self.rb.append(np.array([]))
        data1 = self.rb.get()
        np.testing.assert_array_equal(data0, data1)
        # the same with something in it
        self.rb.append(np.array([0, 1]))
        data0 = self.rb.get()
        self.rb.append(np.array([]))
        data1 = self.rb.get()
        np.testing.assert_array_equal(data0, data1)

    def test_add(self):
        """Various adding variations must work."""
        d0 = np.array([0, 1, 2])
        d1 = np.array([3, 4, 5])
        d2 = np.array([6, 7, 8, 9, 10])
        d3 = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        # non-full-add
        self.rb.append(d0)
        np.testing.assert_array_equal(self.rb.get(), d0)
        # non-full-add
        self.rb.append(d1)
        np.testing.assert_array_equal(self.rb.get(), np.concatenate([d0, d1]))
        # overfull-add
        self.rb.append(d2)
        np.testing.assert_array_equal(self.rb.get(), np.concatenate([d0, d1, d2])[-10:])
        self.rb.append(d3)
        np.testing.assert_array_equal(self.rb.get(), d3[-10:])
        # test overfull add on empty buffer
        d0 = np.arange(11)
        self.rb = RingBuffer(10)
        self.rb.append(d0)
        np.testing.assert_array_equal(self.rb.get(), d0[-10:])
        # test full add on full buffer
        self.rb = RingBuffer(10)
        self.rb.append(np.arange(10))
        d0 = np.arange(10) + 10
        self.rb.append(d0)
        np.testing.assert_array_equal(self.rb.get(), d0)

    def test_add_with_mismatching_dimensions(self):
        """Appending data with mismatching dimension must raise a ValueError."""
        d0 = np.arange(8).reshape(2, 4)
        d1 = np.arange(6).reshape(2, 3)
        self.rb.append(d0)
        with self.assertRaises(ValueError):
            self.rb.append(d1)

if __name__ == '__main__':
    unittest.main()

