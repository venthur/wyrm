
import unittest

import numpy as np

from wyrm.misc import Data
from wyrm.misc import segment_dat


class TestSegmentDat(unittest.TestCase):

    def test_segment_dat(self):
        """Test conversion from Continuous to Epoched data."""
        # create 100 samples and tree channels data
        ones = np.ones((100, 3))
        data = np.array([ones, ones*2, ones*3]).reshape(-1, 3)
        time = np.linspace(0, 3000, 300, endpoint=False)
        channels = ['a', 'b', 'c']
        markers = [[500, 'M1'], [1500, 'M2'], [2500, 'M3']]
        dat = Data(data, [time, channels], ['time', 'channels'], ['ms', '#'])
        dat.markers = markers
        dat.fs = 100
        mrk_def = {'class 1': ['M1'],
                   'class 2': ['M2', 'M3']
                  }
        epo = segment_dat(dat, mrk_def, [-400, 400])
        # test if basic info was transferred from cnt
        self.assertEqual(dat.markers, epo.markers)
        self.assertEqual(dat.fs, epo.fs)
        np.testing.assert_array_equal(dat.axes[-1], epo.axes[-1])
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


if __name__ == '__main__':
    unittest.main()
