import unittest

import numpy as np

from wyrm.misc import Epo
from wyrm.misc import calculate_classwise_average


class TestCalculateClasswiseAverage(unittest.TestCase):

    def test_calculate_classwise_average(self):
        """Calculate classwise average."""
        ones = np.ones((10, 2))
        twoes = ones * 2
        data = np.array([ones, ones, twoes, twoes, ones, twoes, twoes])
        channels = ['c1', 'c2']
        fs = 10
        classes = [0, 0, 1, 1, 0, 1, 1]
        class_names = ['ones', 'twoes']
        markers = []
        ival = [0, 700]
        epo = Epo(data, fs, channels, markers, classes, class_names, ival)
        avg_epo = calculate_classwise_average(epo)
        # check for two epoches (one for each class)
        self.assertEqual(avg_epo.data.shape[0], 2)
        # check if the data is correct
        self.assertEqual(np.average(avg_epo.data[0]), 1)
        self.assertEqual(np.average(avg_epo.data[1]), 2)


if __name__ == '__main__':
    unittest.main()
