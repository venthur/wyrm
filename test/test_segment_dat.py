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

    def test_segment_dat_with_unequally_sized_data(self):
        """Segmentation must ignore too short or too long chunks in the result."""
        # We create a marker that is too close to the beginning of the
        # data, so its cnt will not bee of length [-400, 400] ms. It
        # should not appear in the resulting epo
        self.dat.markers.append([100, 'M1'])
        epo = segment_dat(self.dat, self.mrk_def, [-400, 400])
        self.assertEqual(epo.data.shape[0], 3)

    # the following tests
    # (test_segment_dat_with_restriction_to_new_data_ival...) work very
    # similar but test slightly different conditions. The basic idea is
    # always: we create a small cnt with three markers directly next to
    # each other, the only thing changing between the tests is the
    # interval. We test all possible combinations of the marker position
    # relative to the interval:
    #   [M----], M [---], [--M--], [----M], [---] M
    # we check in each test that with increasing number of new samples
    # the correct number of epochs is returned.
    # WARNING: This is fairly complicated to get right, if you want to
    # change something please make sure you fully understand the problem
    def test_segment_dat_with_restriction_to_new_data_ival_zero_pos(self):
        """Online Segmentation with ival 0..+something must work correctly."""
        # [   0.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800.]
        #         M100                                600
        #                M200                                700
        #                       M299                                799
        #                       M300                                800
        #                       M301                                801
        data = np.ones((9, 3))
        time = np.linspace(0, 900, 9, endpoint=False)
        channels = 'a', 'b', 'c'
        markers = [[100, 'x'], [200, 'x'], [299, 'x'], [300, 'x'], [301, 'x']]
        dat = Data(data, [time, channels], ['time', 'channels'], ['ms', '#'])
        dat.fs = 10
        dat.markers = markers
        mrk_def = {'class 1': ['x']}
        # each tuple has (number of new samples, expected epocs)
        samples_epos = [(0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (5, 5)]
        for s, e in samples_epos:
            epo = segment_dat(dat, mrk_def, [0, 500], newsamples=s)
            self.assertEqual(epo.data.shape[0], e)

    def test_segment_dat_with_restriction_to_new_data_ival_pos_pos(self):
        """Online Segmentation with ival +something..+something must work correctly."""
        # [   0.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800.]
        #         M100    200                         600
        #                M200    300                         700
        #                       M299    399                         799
        #                       M300    400                         800
        #                       M301    401                         801
        data = np.ones((9, 3))
        time = np.linspace(0, 900, 9, endpoint=False)
        channels = 'a', 'b', 'c'
        markers = [[100, 'x'], [200, 'x'], [299, 'x'], [300, 'x'], [301, 'x']]
        dat = Data(data, [time, channels], ['time', 'channels'], ['ms', '#'])
        dat.fs = 10
        dat.markers = markers
        mrk_def = {'class 1': ['x']}
        # each tuple has (number of new samples, expected epocs)
        samples_epos = [(0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (5, 5)]
        for s, e in samples_epos:
            epo = segment_dat(dat, mrk_def, [100, 500], newsamples=s)
            self.assertEqual(epo.data.shape[0], e)

    def test_segment_dat_with_restriction_to_new_data_ival_neg_pos(self):
        """Online Segmentation with ival -something..+something must work correctly."""
        # [   0.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800.]
        #          100                 M400           600
        #                 200                 M500           700
        #                        299                 M599           799
        #                        300                 M600           800
        #                        301                 M601           801
        data = np.ones((9, 3))
        time = np.linspace(0, 900, 9, endpoint=False)
        channels = 'a', 'b', 'c'
        markers = [[400, 'x'], [500, 'x'], [599, 'x'], [600, 'x'], [601, 'x']]
        dat = Data(data, [time, channels], ['time', 'channels'], ['ms', '#'])
        dat.fs = 10
        dat.markers = markers
        mrk_def = {'class 1': ['x']}
        # each tuple has (number of new samples, expected epocs)
        samples_epos = [(0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (5, 5)]
        for s, e in samples_epos:
            epo = segment_dat(dat, mrk_def, [-300, 200], newsamples=s)
            self.assertEqual(epo.data.shape[0], e)

    def test_segment_dat_with_restriction_to_new_data_ival_neg_zero(self):
        """Online Segmentation with ival -something..0 must work correctly."""
        # [   0.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800.]
        #          100                        M500
        #                 200                        M600
        #                        299                        M699
        #                        300                        M700
        #                        301                        M701
        data = np.ones((9, 3))
        time = np.linspace(0, 900, 9, endpoint=False)
        channels = 'a', 'b', 'c'
        markers = [[500, 'x'], [600, 'x'], [699, 'x'], [700, 'x'], [701, 'x']]
        dat = Data(data, [time, channels], ['time', 'channels'], ['ms', '#'])
        dat.fs = 10
        dat.markers = markers
        mrk_def = {'class 1': ['x']}
        # each tuple has (number of new samples, expected epocs)
        samples_epos = [(0, 0), (1, 0), (2, 2), (3, 4), (4, 5), (5, 5)]
        for s, e in samples_epos:
            epo = segment_dat(dat, mrk_def, [-400, 0], newsamples=s)
            self.assertEqual(epo.data.shape[0], e)

    def test_segment_dat_with_restriction_to_new_data_ival_neg_neg(self):
        """Online Segmentation with ival -something..-something must work correctly."""
        # [   0.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800.]
        #          100                  400   M500
        #                 200                  500   M600
        #                        299                  599   M699
        #                        300                  600   M700
        #                        301                  600   M701
        data = np.ones((9, 3))
        time = np.linspace(0, 900, 9, endpoint=False)
        channels = 'a', 'b', 'c'
        markers = [[500, 'x'], [600, 'x'], [699, 'x'], [700, 'x'], [701, 'x']]
        dat = Data(data, [time, channels], ['time', 'channels'], ['ms', '#'])
        dat.fs = 10
        dat.markers = markers
        mrk_def = {'class 1': ['x']}
        # each tuple has (number of new samples, expected epocs)
        samples_epos = [(0, 0), (1, 0), (2, 2), (3, 4), (4, 5), (5, 5)]
        for s, e in samples_epos:
            epo = segment_dat(dat, mrk_def, [-400, -100], newsamples=s)
            self.assertEqual(epo.data.shape[0], e)

    def test_segment_dat_with_negative_newsamples(self):
        """Raise an error when newsamples is not positive or None."""
        with self.assertRaises(AssertionError):
            segment_dat(self.dat, self.mrk_def, [-400, 400], newsamples=-1)

    def test_segment_dat_copy(self):
        """segment_dat must not modify arguments."""
        cpy = self.dat.copy()
        segment_dat(self.dat, self.mrk_def, [-400, 400])
        self.assertEqual(cpy, self.dat)

    def test_segment_dat_swapaxes(self):
        """Segmentation must work with nonstandard axes."""
        epo = segment_dat(swapaxes(self.dat, 0, 1), self.mrk_def, [-400, 400], timeaxis=-1)
        # segment_dat added a new dimension
        epo = swapaxes(epo, 1, 2)
        epo2 = segment_dat(self.dat, self.mrk_def, [-400, 400])
        self.assertEqual(epo, epo2)

    def test_equivalent_axes(self):
        """Segmentation must deal with equivalent axis indices correctly."""
        epo0 = segment_dat(self.dat, self.mrk_def, [-400, 400], timeaxis=-2)
        epo1 = segment_dat(self.dat, self.mrk_def, [-400, 400], timeaxis=0)
        self.assertEqual(epo0, epo1)

if __name__ == '__main__':
    unittest.main()
