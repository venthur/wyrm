#!/usr/bin/env python


import unittest

import numpy as np

from wyrm.types import RingBuffer, Data


def data_factory(data, axes=None, names=None, units=None, markers=None):
    """Helper method to create Data objects."""
    if len(data) == 0:
        axes = names = units = []
    else:
        if axes is None:
            axes = []
            for i in range(data.ndim):
                a = [i * 10 for i in range(data.shape[i])]
                axes.append(a)
        if names is None:
            names = ['name %i' % i for i in range(data.ndim)]
        if units is None:
            units = ['unit %i' % i for i in range(data.ndim)]
    d = Data(data=data, axes=axes, names=names, units=units)
    d.markers = markers if markers is not None else []
    d.fs = 100
    return d


class TestRingBuffer(unittest.TestCase):

    def setUp(self):
        self.rb = RingBuffer(100)

    def test_add_empty(self):
        """Adding an emtpy array must not modify the ringbuffer."""
        # start with emtpy rb
        empty = data_factory(np.array([]))
        dat0 = self.rb.get()
        self.rb.append(empty)
        dat1 = self.rb.get()
        self.assertEqual(dat0, dat1)
        # the same with something in it
        small = data_factory(np.array([0, 1]))
        self.rb.append(small)
        dat0 = self.rb.get()
        self.rb.append(empty)
        dat1 = self.rb.get()
        self.assertEqual(dat0, dat1)

    def test_add(self):
        """Various adding variations must work."""
        d0 = data_factory(np.array([0, 1, 2]))
        d1 = data_factory(np.array([3, 4, 5]))
        d2 = data_factory(np.array([6, 7, 8, 9, 10]))
        d3 = data_factory(np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]))
        # non-full-add
        self.rb.append(d0)
        data0 = self.rb.get()
        self.assertEqual(data0, d0)
        # non-full-add
        self.rb.append(d1)
        data1 = self.rb.get()
        self.assertEqual(data1, data_factory(np.concatenate([d0.data, d1.data])))
        # overfull-add
        self.rb.append(d2)
        data2 = self.rb.get()
        self.assertEqual(data2, data_factory(np.concatenate([d0.data, d1.data, d2.data])[-10:]))
        self.rb.append(d3)
        data3 = self.rb.get()
        self.assertEqual(data3, data_factory(d3.data[-10:]))
        # test overfull add on empty buffer
        d0 = data_factory(np.arange(11))
        self.rb = RingBuffer(100)
        self.rb.append(d0)
        data0 = self.rb.get()
        self.assertEqual(data0, data_factory(d0.data[-10:]))
        # test full add on full buffer
        self.rb = RingBuffer(100)
        self.rb.append(data_factory(np.arange(10)))
        d0 = data_factory(np.arange(10) + 10)
        self.rb.append(d0)
        data0 = self.rb.get()
        self.assertEqual(data0, d0)

    def test_add_with_mismatching_dimensions(self):
        """Appending data with mismatching dimension must raise a ValueError."""
        d0 = data_factory(np.arange(8).reshape(2, 4))
        d1 = data_factory(np.arange(6).reshape(2, 3))
        self.rb.append(d0)
        with self.assertRaises(ValueError):
            self.rb.append(d1)

    def test_add_with_markers(self):
        # add three elements to empty buffer pluss three markers
        m = [[0, '0'], [20, '2']]
        d3 = data_factory(np.arange(9).reshape(3, 3))
        d3_w_markers = d3.copy(markers=m)
        self.rb.append(d3_w_markers)
        data = self.rb.get()
        self.assertEqual(data.markers, m)
        # move three more elements into the buffer, the markers should
        # stay the same, as we don't overfilled the buffer yet
        self.rb.append(d3)
        data = self.rb.get()
        self.assertEqual(data.markers, m)
        # now we do it again twice and the first one should disappear
        # the second one should move one position back
        self.rb.append(d3)
        self.rb.append(d3)
        data = self.rb.get()
        self.assertEqual(data.markers, [[0, '2']])
        # again, and no markers should be left
        self.rb.append(d3)
        data = self.rb.get()
        self.assertEqual(data.markers, [])
        m = [[0, '0'], [90, '9'], [100, '10']]
        d11_w_markers = data_factory(np.arange(33).reshape(11, 3), markers=m)
        self.rb.append(d11_w_markers)
        data = self.rb.get()
        self.assertEqual(data.markers, [[80, '9'], [90, '10']])
        # test overfull add on empty buffer
        m0 = [[i * 10, i] for i in range(11)]
        d0_w_markers = data_factory(np.arange(33).reshape(11, 3), markers=m0)
        self.rb = RingBuffer(100)
        self.rb.append(d0_w_markers)
        data = self.rb.get()
        self.assertEqual(data.markers, map(lambda x: [x[0]-10, x[1]], m0[-10:]))
        # test full add on full buffer
        self.rb = RingBuffer(100)
        m0 = [[i, i] for i in range(10)]
        self.rb.append(data_factory(np.arange(30).reshape(10, 3), markers=m0))
        m0 = map(lambda x: [x[0], x[1]+10], m0)
        self.rb.append(data_factory(np.arange(30).reshape(10, 3), markers=m0))
        data = self.rb.get()
        self.assertEqual(data.markers, m0)

    def test_appending_until_twice_full_w_1hz(self):
        """Test appending until twice full with 1Hz."""
        d = data_factory(np.array([0]))
        d.fs = 1
        rb = RingBuffer(10000)
        try:
            for i in range(21):
                rb.append(d)
                rb.get()
        except AssertionError:
            self.fail()

    def test_different_lengths(self):
        """Check if setting the length works correctly."""
        d = data_factory(np.array([1]))
        testdata = [
                # ms, fs, elems
                [1000, 1000, 1000],
                [1000, 100, 100],
                [1000, 10, 10],
                [1000, 1, 1],
                [2000, 1000, 2000],
                [2000, 100, 200],
                [2000, 10, 20],
                [2000, 1, 2],
                [1000, 2000, 2000],
                [1000, 200, 200],
                [1000, 20, 20],
                [1000, 2, 2],
                [1000, 1000, 1000],
                [100, 1000, 100],
                [10, 1000, 10],
                [1, 1000, 1],
                 ]
        # 1000ms / 1000hz -> 1000 elements
        for ms, fs, elems in testdata:
            rb = RingBuffer(ms)
            d.fs = fs
            rb.append(d)
            self.assertEqual(rb.length, elems)

if __name__ == '__main__':
    unittest.main()

