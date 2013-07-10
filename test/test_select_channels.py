import unittest

import numpy as np

from wyrm.misc import Cnt
from wyrm.misc import select_channels


class TestSelectChannels(unittest.TestCase):

    def setUp(self):
        raw = np.arange(20).reshape(4, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        fs = 10
        marker = []
        self.cnt = Cnt(raw, fs, channels, marker)

    def test_select_channels(self):
        """Selecting channels with an array of regexes."""
        channels = self.cnt.data.copy()
        self.cnt = select_channels(self.cnt, ['ca.*', 'cc1'])
        np.testing.assert_array_equal(self.cnt.channels,  np.array(['ca1', 'ca2', 'cc1']))
        np.testing.assert_array_equal(self.cnt.data, channels[:, np.array([0, 1, -1])])

    def test_select_channels_inverse(self):
        """Removing channels with an array of regexes."""
        channels = self.cnt.data.copy()
        self.cnt = select_channels(self.cnt, ['ca.*', 'cc1'], invert=True)
        np.testing.assert_array_equal(self.cnt.channels,  np.array(['cb1', 'cb2']))
        np.testing.assert_array_equal(self.cnt.data, channels[:, np.array([2, 3])])


if __name__ == '__main__':
    unittest.main()
