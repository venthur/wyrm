from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data, BlockBuffer
from wyrm.processing import append_cnt


class TestBlockBuffer(unittest.TestCase):

    def setUp(self):
        self.empty_dat = Data(np.array([]), [], [], [])
        data = np.array([0, 0])
        self.dat_1 = Data(np.array([0, 0])[np.newaxis, :], [np.array([0]), np.array(['ch1', 'ch2'])], ['time', 'channel'], ['ms', '#'])
        self.dat_1.fs = 1000
        self.dat_1.markers = [[0, 'x']]
        self.dat_5 = reduce(append_cnt, [self.dat_1 for i in range(5)])

    def test_append_empty(self):
        """Appending several emtpy dats must not modify the Block Buffer."""
        b = BlockBuffer(5)
        b.append(self.empty_dat)
        b.append(self.empty_dat)
        b.append(self.empty_dat)
        b.append(self.empty_dat)
        b.append(self.empty_dat)
        b.append(self.empty_dat)
        self.assertEqual(self.empty_dat, b.get())

    def test_append_until_full(self):
        """Appending fractions of block_length, must accumulate in the buffer until block_length is reached."""
        b = BlockBuffer(5)
        for i in range(4):
            b.append(self.dat_1)
        ret = b.get()
        self.assertEqual(self.empty_dat, ret)
        b.append(self.dat_1)
        ret = b.get()
        self.assertEqual(self.dat_5, ret)

    def test_append_with_markers(self):
        """Check if markers are handled correctly."""
        markers = [[i, 'x'] for i in range(5)]
        b = BlockBuffer(5)
        for i in range(4):
            b.append(self.dat_1)
        ret = b.get()
        self.assertEqual(self.empty_dat, ret)
        b.append(self.dat_1)
        ret = b.get()
        self.assertEqual(ret.markers, markers)


if __name__ == '__main__':
    unittest.main()
