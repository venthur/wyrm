from __future__ import division

import unittest

import random
import numpy as np

from wyrm.types import Data
from wyrm.processing import sort_channels, swapaxes, CHANNEL_10_20


class TestSortChannels(unittest.TestCase):

    def setUp(self):
        self.sorted_channels = np.array([name for name, pos in CHANNEL_10_20])
        channels = self.sorted_channels.copy()
        random.shuffle(channels)
        raw = np.random.random((5, 10, len(channels)))
        time = np.linspace(0, 1000, 10, endpoint=False)
        epochs = np.array([0, 1, 0, 1, 0])
        fs = 100
        marker = [[100, 'foo'], [200, 'bar']]
        self.dat = Data(raw, [epochs, time, channels], ['class', 'time', 'channels'], ['#', 'ms', '#'])
        self.dat.fs = fs
        self.dat.markers = marker

    def test_sort_channels(self):
        """sort_channels must sort correctly."""
        dat = sort_channels(self.dat)
        np.testing.assert_array_equal(dat.axes[-1], self.sorted_channels)

    def test_sort_channels_with_unknown_channel(self):
        """Unknown channels move to the back."""
        self.dat.axes[-1][7] = 'XX'
        dat = sort_channels(self.dat)
        self.assertEqual(dat.axes[-1][-1], 'XX')

    def test_sort_channels_swapaxis(self):
        """sort_channels must workt with nonstandard chanaxis."""
        sorted_ = sort_channels(swapaxes(self.dat, 1, -1), 1)
        sorted_ = swapaxes(sorted_, 1, -1)
        sorted2 = sort_channels(self.dat)
        self.assertEqual(sorted_, sorted2)

    def test_sort_channels_copy(self):
        """sort_channels must not modify argument."""
        cpy = self.dat.copy()
        sort_channels(self.dat)
        self.assertEqual(self.dat, cpy)

if __name__ == '__main__':
    unittest.main()
