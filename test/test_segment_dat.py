from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import segment_dat
from wyrm.processing import swapaxes


class TestSegmentDat(unittest.TestCase):

    def setUp(self):
        # create 100 samples and tree channels data
        ones = np.ones((100, 3))
        data = np.array([ones, ones*2, ones*3]).reshape(-1, 3)
        time = np.linspace(0, 3000, 300, endpoint=False)
        channels = ['a', 'b', 'c']
        markers = [[500, 'M1'], [1500, 'M2'], [2500, 'M3']]
        self.dat = Data(data, [time, channels], ['time', 'channels'], ['ms', '#'])
        self.dat.markers = markers
        self.dat.fs = 100
        self.mrk_def = {'class 1': ['M1'],
                        'class 2': ['M2', 'M3']
                       }

    def test_segment_dat(self):
        """Test conversion from Continuous to Epoched data."""
        epo = segment_dat(self.dat, self.mrk_def, [-400, 400])
        # test if basic info was transferred from cnt
        self.assertEqual(self.dat.markers, epo.markers)
        self.assertEqual(self.dat.fs, epo.fs)
        np.testing.assert_array_equal(self.dat.axes[-1], epo.axes[-1])
        # test if the actual data is correct
        self.assertEqual(list(epo.axes[0]), [0, 1, 1])
        np.testing.assert_array_equal(epo.class_names, np.array(['class 1', 'class 2']))
        self.assertEqual(epo.data.shape, (3, 80, 3))
        for i in range(3):
            e = epo.data[i, ...]
            self.assertEqual(np.average(e), i+1)
        # test if the epo.ival is the same we cut out
        self.assertEqual(epo.axes[-2][0], -400)
        self.assertEqual(epo.axes[-2][-1], 390)

    def test_segment_dat_with_nonexisting_markers(self):
        """Segmentation without result should return empty .data"""
        mrk_def = {'class 1': ['FUU1'],
                   'class 2': ['FUU2', 'FUU3']
                  }
        epo = segment_dat(self.dat, mrk_def, [-400, 400])
        self.assertEqual(epo.data.shape[0], 0)

    def test_segment_dat_copy(self):
        """segment_dat must not modify arguments."""
        cpy = self.dat.copy()
        segment_dat(self.dat, self.mrk_def, [-400, 400])
        self.assertEqual(cpy, self.dat)

    def test_segment_dat_swapaxes(self):
        epo = segment_dat(swapaxes(self.dat, 0, 1), self.mrk_def, [-400, 400], timeaxis=-1)
        # segment_dat added a new dimension
        epo = swapaxes(epo, 1, 2)
        epo2 = segment_dat(self.dat, self.mrk_def, [-400, 400])
        self.assertEqual(epo, epo2)

if __name__ == '__main__':
    unittest.main()
