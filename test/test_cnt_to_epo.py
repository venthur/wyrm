
import unittest

import numpy as np

from wyrm.misc import Cnt
from wyrm.misc import Epo
from wyrm.misc import cnt_to_epo


class TestCntToEpo(unittest.TestCase):

    def test_cnt_to_epo(self):
        """Test conversion from Continuous to Epoched data."""
        # create 100 samples and tree channels data
        ones = np.ones((100, 3))
        data = np.array([ones, ones*2, ones*3]).reshape(-1, 3)
        fs = 100
        channels = ['a', 'b', 'c']
        markers = [[49, 'M1'], ['149', 'M2'], ['249', 'M3']]
        cnt = Cnt(data, fs, channels, markers)
        mrk_def = {'class 1': ['M1'],
                   'class 2': ['M2', 'M3']
                  }
        epo = cnt_to_epo(cnt, mrk_def, [-400, 400])
        # test if basic info was transferred from cnt
        self.assertEqual(fs, epo.fs)
        self.assertEqual(channels, list(epo.channel))
        # test if the actual data is correct
        self.assertEqual(list(epo.classes), [0, 1, 1])
        self.assertEqual(epo.class_names, ['class 1', 'class 2'])
        self.assertEqual(epo.data.shape, (3, 80, 3))
        for i in range(3):
            e = epo.data[i, ...]
            self.assertEqual(np.average(e), i+1)


if __name__ == '__main__':
    unittest.main()
