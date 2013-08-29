from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import select_channels
from wyrm.processing import swapaxes

class TestSelectChannels(unittest.TestCase):

    def setUp(self):
        raw = np.arange(20).reshape(4, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.arange(4)
        self.dat = Data(raw, [time, channels], ['time', 'channels'], ['ms', '#'])

    def test_select_channels(self):
        """Selecting channels with an array of regexes."""
        channels = self.dat.data.copy()
        self.dat = select_channels(self.dat, ['ca.*', 'cc1'])
        np.testing.assert_array_equal(self.dat.axes[-1],  np.array(['ca1', 'ca2', 'cc1']))
        np.testing.assert_array_equal(self.dat.data, channels[:, np.array([0, 1, -1])])

    def test_select_channels_inverse(self):
        """Removing channels with an array of regexes."""
        channels = self.dat.data.copy()
        self.dat = select_channels(self.dat, ['ca.*', 'cc1'], invert=True)
        np.testing.assert_array_equal(self.dat.axes[-1],  np.array(['cb1', 'cb2']))
        np.testing.assert_array_equal(self.dat.data, channels[:, np.array([2, 3])])

    def test_select_channels_copy(self):
        """Select channels must not change the original parameter."""
        cpy = self.dat.copy()
        select_channels(self.dat, ['ca.*'])
        self.assertEqual(cpy, self.dat)

    def test_select_channels_swapaxis(self):
        """Select channels works with non default chanaxis."""
        dat1 = select_channels(swapaxes(self.dat, 0, 1), ['ca.*'], chanaxis=0)
        dat1 = swapaxes(dat1, 0, 1)
        dat2 = select_channels(self.dat, ['ca.*'])
        self.assertEqual(dat1, dat2)

if __name__ == '__main__':
    unittest.main()
