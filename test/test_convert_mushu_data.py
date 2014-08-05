from __future__ import division

import unittest

import numpy as np

from wyrm.io import convert_mushu_data


CHANNELS = 5
SAMPLES = 3
FS = 10


class TestConvertMushuData(unittest.TestCase):

    def setUp(self):
        xv, yv = np.meshgrid(list(range(CHANNELS)), list(range(SAMPLES)))
        self.data = xv * 10 + yv
        self.channels = ['ch %i' % i for i in range(CHANNELS)]
        self.time = np.linspace(0, SAMPLES / FS * 1000, SAMPLES, endpoint=False)
        self.markers = [[-10, 'a'], [0, 'b'], [1e100000, 'c']]

    def test_convert_mushu_data(self):
        """Convert mushu data."""
        cnt = convert_mushu_data(self.data, self.markers, FS, self.channels)
        # data
        np.testing.assert_array_equal(cnt.data, self.data)
        self.assertEqual(cnt.data.shape, (SAMPLES, CHANNELS))
        # axes
        timeaxis = np.linspace(0, SAMPLES / FS * 1000, SAMPLES, endpoint=False)
        np.testing.assert_array_equal(cnt.axes[0], timeaxis)
        np.testing.assert_array_equal(cnt.axes[1], self.channels)
        # names and units
        self.assertEqual(cnt.names, ['time', 'channel'])
        self.assertEqual(cnt.units, ['uV', '#'])
        # fs
        self.assertEqual(cnt.fs, FS)
        # markers
        self.assertEqual(cnt.markers, self.markers)

    def test_convert_mushu_data_copy(self):
        """convert_mushu_data should make a copy of its arguments."""
        data = self.data.copy()
        channels = self.channels[:]
        markers = self.markers[:]
        cnt = convert_mushu_data(data, self.markers, FS, channels)
        # data
        data[0, 0] = -1
        np.testing.assert_array_equal(cnt.data, self.data)
        # axes
        channels[0] = 'FOO'
        np.testing.assert_array_equal(cnt.axes[-1], self.channels)
        # markers
        markers[0][0] = markers[0][0] - 1
        self.assertEqual(cnt.markers, self.markers)

if __name__ == '__main__':
    unittest.main()
