from __future__ import division

import unittest

import numpy as np
np.random.seed(42)

from wyrm.types import Data
from wyrm.processing import calculate_csp
from wyrm.processing import swapaxes


class TestCalculateCSP(unittest.TestCase):

    EPOCHS = 50
    SAMPLES = 100
    SOURCES = 2
    CHANNELS = 10

    def setUp(self):
        # create a random noise signal with 50 epochs, 100 samples, and
        # 2 sources
        # even epochs and source 0: *= 5
        # odd epochs and source 1: *= 5
        self.s = np.random.randn(self.EPOCHS, self.SAMPLES, self.SOURCES)
        self.s[ ::2, :, 0] *= 5
        self.s[1::2, :, 1] *= 5
        # the mixmatrix which converts our sources to channels
        # X = As + noise
        self.A = np.random.randn(self.CHANNELS, self.SOURCES)
        # our 'signal' which 50 epochs, 100 samples and 10 channels
        self.X = np.empty((self.EPOCHS, self.SAMPLES, self.CHANNELS))
        for i in range(self.EPOCHS):
            self.X[i] = np.dot(self.A, self.s[i].T).T
        noise = np.random.randn(self.EPOCHS, self.SAMPLES, self.CHANNELS) * 0.01
        self.X += noise

    def test_d(self):
        """Test if the first lambda is almost 1 and the last one almost -1."""
        W, A_est, d = calculate_csp(self.X[::2], self.X[1::2])
        epsilon = 0.1
        self.assertAlmostEqual(d[0], 1, delta=epsilon)
        self.assertAlmostEqual(d[-1], -1, delta=epsilon)

    def test_A(self):
        """Test if A_est is elementwise almost equal A."""
        W, A_est, d = calculate_csp(self.X[::2], self.X[1::2])
        # A and A_est can have a different scaling, after normalizing
        # and correcting for sign, they should be almost equal
        # normalize (we're only interested in the first and last column)
        for i in 0, -1:
            idx = np.argmax(np.abs(A_est[:, i]))
            A_est[:, i] /= A_est[idx, i]
            idx = np.argmax(np.abs(self.A[:, i]))
            self.A[:, i] /= self.A[idx, i]
        # for i in 0, -1:
        #   check elementwise if A[:, i] almost A_est[:, i]
        epsilon = 0.01
        for i in 0, -1:
            diff = self.A[:, i] - A_est[:, i]
            diff = np.abs(diff)
            diff = np.sum(diff) / self.A.shape[0]
            self.assertTrue(diff < epsilon)

    def test_s(self):
        """Test if s_est is elementwise almost equal s."""
        W, A_est, d = calculate_csp(self.X[::2], self.X[1::2])
        # applying the filter to X gives us s_est which should be almost
        # equal s
        s_est = np.empty(self.s.shape)
        for i in range(self.EPOCHS):
            s_est[i] = np.dot(self.X[i], W[:, [0, -1]])
        # correct for scaling, and sign
        self.s = self.s.reshape(-1, self.SOURCES)
        s_est2 = s_est.reshape(-1, self.SOURCES)
        epsilon = 0.01
        for i in range(self.SOURCES):
            idx = np.argmax(np.abs(s_est2[:, i]))
            s_est2[:, i] /= s_est2[idx, i]
            idx = np.argmax(np.abs(self.s[:, i]))
            self.s[:, i] /= self.s[idx, i]
            diff = np.sum(np.abs(self.s[:, i] - s_est2[:, i])) / self.s.shape[0]
            self.assertTrue(diff < epsilon)

    #def test_calculate_signed_r_square_swapaxes(self):
    #    """caluclate_r_square must work with nonstandard classaxis."""
    #    dat = calculate_signed_r_square(swapaxes(self.dat, 0, 2), classaxis=2)
    #    # the class-axis just dissapears during
    #    # calculate_signed_r_square, so axis 2 becomes axis 1
    #    dat = dat.swapaxes(0, 1)
    #    dat2 = calculate_signed_r_square(self.dat)
    #    np.testing.assert_array_equal(dat, dat2)

    #def test_calculate_signed_r_square_copy(self):
    #    """caluclate_r_square must not modify argument."""
    #    cpy = self.dat.copy()
    #    calculate_signed_r_square(self.dat)
    #    self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
