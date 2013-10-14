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
        data0, _ = self.rb.get()
        self.rb.append(np.array([]))
        data1, _ = self.rb.get()
        np.testing.assert_array_equal(data0, data1)
        # the same with something in it
        self.rb.append(np.array([0, 1]))
        data0, _ = self.rb.get()
        self.rb.append(np.array([]))
        data1, _ = self.rb.get()
        np.testing.assert_array_equal(data0, data1)

    def test_add(self):
        """Various adding variations must work."""
        d0 = np.array([0, 1, 2])
        d1 = np.array([3, 4, 5])
        d2 = np.array([6, 7, 8, 9, 10])
        d3 = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        # non-full-add
        self.rb.append(d0)
        data0, _ = self.rb.get()
        np.testing.assert_array_equal(data0, d0)
        # non-full-add
        self.rb.append(d1)
        data1, _ = self.rb.get()
        np.testing.assert_array_equal(data1, np.concatenate([d0, d1]))
        # overfull-add
        self.rb.append(d2)
        data2, _ = self.rb.get()
        np.testing.assert_array_equal(data2, np.concatenate([d0, d1, d2])[-10:])
        self.rb.append(d3)
        data3, _ = self.rb.get()
        np.testing.assert_array_equal(data3, d3[-10:])
        # test overfull add on empty buffer
        d0 = np.arange(11)
        self.rb = RingBuffer(10)
        self.rb.append(d0)
        data0, _ = self.rb.get()
        np.testing.assert_array_equal(data0, d0[-10:])
        # test full add on full buffer
        self.rb = RingBuffer(10)
        self.rb.append(np.arange(10))
        d0 = np.arange(10) + 10
        self.rb.append(d0)
        data0, _ = self.rb.get()
        np.testing.assert_array_equal(data0, d0)

    def test_add_with_mismatching_dimensions(self):
        """Appending data with mismatching dimension must raise a ValueError."""
        d0 = np.arange(8).reshape(2, 4)
        d1 = np.arange(6).reshape(2, 3)
        self.rb.append(d0)
        with self.assertRaises(ValueError):
            self.rb.append(d1)

    def test_add_with_markers(self):
        # add three elements to empty buffer pluss three markers
        d3 = np.arange(3)
        m = [[0, '0'], [2, '2']]
        self.rb.append(d3, m)
        _, m_ = self.rb.get()
        self.assertEqual(m_, m)
        # move three more elements into the buffer, the markers should
        # stay the same, as we don't overfilled the buffer yet
        self.rb.append(d3)
        _, m_ = self.rb.get()
        self.assertEqual(m_, m)
        # now we do it again twice and the first one should disappear
        # the second one should move one position back
        self.rb.append(d3)
        self.rb.append(d3)
        _, m_ = self.rb.get()
        self.assertEqual(m_, [[0, '2']])
        # again, and no markers should be left
        self.rb.append(d3)
        _, m_ = self.rb.get()
        self.assertEqual(m_, [])
        d11 = np.arange(11)
        m = [[0, '0'], [9, '9'], [10, '10']]
        self.rb.append(d11, m)
        _, m_ = self.rb.get()
        self.assertEqual(m_, [[8, '9'], [9, '10']])
        # test overfull add on empty buffer
        d0 = np.arange(11)
        m0 = [[i, i] for i in range(11)]
        self.rb = RingBuffer(10)
        self.rb.append(d0, m0)
        _, marker0 = self.rb.get()
        self.assertEqual(marker0, marker0[-10:])
        # test full add on full buffer
        self.rb = RingBuffer(10)
        m0 = [[i, i] for i in range(10)]
        self.rb.append(np.arange(10), m0)
        m0 = map(lambda x: [x[0], x[1]+10], m0)
        self.rb.append(np.arange(10), m0)
        _, marker0 = self.rb.get()
        self.assertEqual(marker0, m0)


if __name__ == '__main__':
    unittest.main()

