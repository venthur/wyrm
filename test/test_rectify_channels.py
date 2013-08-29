from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import rectify_channels

class TestRectifytChannels(unittest.TestCase):

    def setUp(self):
        raw = np.arange(20).reshape(4, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.arange(4)
        self.dat = Data(raw, [time, channels], ['time', 'channels'], ['ms', '#'])

    def test_rectify_channels(self):
        """Rectify channels of positive and negative data must be equal."""
        dat = rectify_channels(self.dat.copy(data=-self.dat.data))
        dat2 = rectify_channels(self.dat)
        self.assertEqual(dat, dat2)

    def test_rectify_channels_copy(self):
        """Rectify channels must not change the original parameter."""
        cpy = self.dat.copy()
        rectify_channels(self.dat)
        self.assertEqual(cpy, self.dat)

if __name__ == '__main__':
    unittest.main()
