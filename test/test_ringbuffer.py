#!/usr/bin/env python


import unittest

import numpy as np

from wyrm.types import RingBuffer


class TestRingBuffer(unittest.TestCase):

    def setUp(self):
        self.rb = RingBuffer((10, ))

    def test_add_empty(self):
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
        d0 = np.array([0, 1, 2])
        d1 = np.array([3, 4, 5])
        d2 = np.array([6, 7, 8, 9, 10])
        d3 = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        self.rb.append(d0)
        np.testing.assert_array_equal(self.rb.get(), d0)
        self.rb.append(d1)
        np.testing.assert_array_equal(self.rb.get(), np.concatenate([d0, d1]))
        self.rb.append(d2)
        np.testing.assert_array_equal(self.rb.get(), np.concatenate([d0, d1, d2])[-10:])
        self.rb.append(d3)
        np.testing.assert_array_equal(self.rb.get(), d3[-10:])

    def test_add2(self):
        d0 = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        self.rb.append(d0)
        np.testing.assert_array_equal(self.rb.get(), d0[-10:])

    def test_add3(self):
        d0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.rb.append(d0)
        np.testing.assert_array_equal(self.rb.get(), d0)
        d1 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        self.rb.append(d1)
        np.testing.assert_array_equal(self.rb.get(), d1)


if __name__ == '__main__':
    unittest.main()

