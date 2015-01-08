from __future__ import division

import unittest

import numpy as np
from numpy.random import randn
np.random.seed(42)

from wyrm.types import Data
from wyrm.processing import append, swapaxes, calculate_cca


class TestCalculateCCA(unittest.TestCase):

    SAMPLES = 1000
    CHANNELS_X = 10
    CHANNELS_Y = 5
    NOISE_LEVEL = 0.1

    def setUp(self):
        # X is a random mixture matrix of random variables
        Sx = randn(self.SAMPLES, self.CHANNELS_X)
        Ax = randn(self.CHANNELS_X, self.CHANNELS_X)
        self.X = np.dot(Sx, Ax)
        # Y is a random mixture matrix of random variables except the
        # first component
        Sy = randn(self.SAMPLES, self.CHANNELS_Y)
        Sy[:, 0] = Sx[:, 0] + self.NOISE_LEVEL * randn(self.SAMPLES)
        Ay = randn(self.CHANNELS_Y, self.CHANNELS_Y)
        self.Y = np.dot(Sy, Ay)
        # generate Data object
        axes_x = [np.arange(self.X.shape[0]), np.arange(self.X.shape[1])]
        axes_y = [np.arange(self.Y.shape[0]), np.arange(self.Y.shape[1])]
        self.dat_x = Data(self.X, axes=axes_x, names=['time', 'channel'], units=['ms', '#'])
        self.dat_y = Data(self.Y, axes=axes_y, names=['time', 'channel'], units=['ms', '#'])

    def test_rho(self):
        """Test if the canonical correlation coefficient almost equals 1."""
        rho, w_x, w_y = calculate_cca(self.dat_x, self.dat_y)
        self.assertAlmostEqual(rho, 1.0, delta=0.01)

    def test_diff_between_canonical_variables(self):
        """Test if the scaled canonical variables are almost same."""
        rho, w_x, w_y = calculate_cca(self.dat_x, self.dat_y)
        cv_x = np.dot(self.X, w_x)
        cv_y = np.dot(self.Y, w_y)

        def scale(x):
            tmp = x - x.mean()
            return tmp / tmp[np.argmax(np.abs(tmp))]

        diff = scale(cv_x) - scale(cv_y)
        diff = np.sum(np.abs(diff)) / self.SAMPLES
        self.assertTrue(diff < 0.1)

    def test_raise_error_with_non_continuous_data(self):
        """Raise error if ``dat_x`` is not continuous Data object."""
        dat = Data(randn(2, self.SAMPLES, self.CHANNELS_X),
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

    def test_calculate_cca_swapaxes(self):
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
