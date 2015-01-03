from __future__ import division

import unittest

import numpy as np
np.random.seed(42)

from wyrm.types import Data
from wyrm.processing import append, swapaxes, calculate_cca


class TestCalculateCCA(unittest.TestCase):

    SAMPLES = 100
    CHANNELS_X = 10
    CHANNELS_Y = 5

    def setUp(self):
        # create a random noise signals with
        # X: 100 samplse and 10 channels
        # Y: 100 samples and 5 channels
        self.X = np.random.randn(self.SAMPLES, self.CHANNELS_X)
        self.Y = np.random.randn(self.SAMPLES, self.CHANNELS_Y)
        axes_x = [np.arange(self.X.shape[0]), np.arange(self.X.shape[1])]
        axes_y = [np.arange(self.Y.shape[0]), np.arange(self.Y.shape[1])]
        self.dat_x = Data(self.X, axes=axes_x, names=['time', 'channel'], units=['ms', '#'])
        self.dat_y = Data(self.Y, axes=axes_y, names=['time', 'channel'], units=['ms', '#'])

    def test_rho(self):
        """Test if the canonical correlation is between 0 and 1."""
        rho, w_x, w_y = calculate_cca(self.dat_x, self.dat_y)
        self.assertTrue(0 < rho <1)

    def test_raise_error_with_non_continuous_data(self):
        """Raise error if ``dat_x`` is not continuous Data object."""
        dat = Data(np.random.randn(2, self.SAMPLES, self.CHANNELS_X),
                   axes=[[0, 1], self.dat_x.axes[0], self.dat_x.axes[1]],
                   names=['class', 'time', 'channel'],
                   units=['#', 'ms', '#'])
        with self.assertRaises(AssertionError):
            calculate_cca(dat, self.dat_x)

    def test_raise_error_with_different_length_data(self):
        """Raise error if the length of ``dat_x`` and ``dat_y`` is different."""
        dat = append(self.dat_x, self.dat_x)
        with self.assertRaises(AssertionError):
            calculate_cca(dat, self.dat_y)

    def test_calculate_csp_swapaxes(self):
        """caluclate_cca must work with nonstandard timeaxis."""
        res1 = calculate_cca(swapaxes(self.dat_x, 0, 1), swapaxes(self.dat_y, 0, 1), timeaxis=1)
        res2 = calculate_cca(self.dat_x, self.dat_y)
        np.testing.assert_array_equal(res1[0], res2[0])
        np.testing.assert_array_equal(res1[1], res2[1])
        np.testing.assert_array_equal(res1[2], res2[2])

    def test_calculate_cca_copy(self):
        """caluclate_cca must not modify argument."""
        cpy_x = self.dat_x.copy()
        cpy_y = self.dat_y.copy()
        calculate_cca(self.dat_x, self.dat_y)
        self.assertEqual(self.dat_x, cpy_x)
        self.assertEqual(self.dat_y, cpy_y)


if __name__ == '__main__':
    unittest.main()
