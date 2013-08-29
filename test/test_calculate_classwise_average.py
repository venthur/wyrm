from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import calculate_classwise_average


class TestCalculateClasswiseAverage(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 2))
        twoes = ones * 2
        # 7 epochs
        data = np.array([ones, ones, twoes, twoes, ones, twoes, twoes])
        channels = ['c1', 'c2']
        time = np.linspace(0, 1000, 10)
        classes = [0, 0, 1, 1, 0, 1, 1]
        class_names = ['ones', 'twoes']
        self.dat = Data(data, [classes, time, channels], ['class', 'time', 'channel'], ['#', 'ms', '#'])
        self.dat.class_names = class_names

    def test_calculate_classwise_average(self):
        """Calculate classwise average."""
        avg_dat = calculate_classwise_average(self.dat)
        # check for two datches (one for each class)
        self.assertEqual(avg_dat.data.shape[0], 2)
        # check if the data is correct
        self.assertEqual(np.average(avg_dat.data[0]), 1)
        self.assertEqual(np.average(avg_dat.data[1]), 2)
        # check if we have as many classes on axes as we have in data
        self.assertEqual(avg_dat.data.shape[0], len(avg_dat.axes[0]))
        #
        self.assertEqual(avg_dat.class_names, self.dat.class_names)

    def test_calculate_classwise_average_with_cnt(self):
        """Calculate classwise avg must raise an error if called with continouos data."""
        del(self.dat.class_names)
        with self.assertRaises(AssertionError):
            calculate_classwise_average(self.dat)

    def test_calculate_classwise_average_copy(self):
        """Calculate classwise avg must not modify the argument."""
        cpy = self.dat.copy()
        calculate_classwise_average(self.dat)
        self.assertEqual(self.dat, cpy)

if __name__ == '__main__':
    unittest.main()
