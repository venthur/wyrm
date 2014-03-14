from __future__ import division

import unittest

import numpy as np
np.random.seed(42)

from wyrm.types import Data
from wyrm.processing import calculate_spoc


class TestCalculateSpoc(unittest.TestCase):

    EPOCHS = 50
    SAMPLES = 100
    SOURCES = 2
    CHANNELS = 10

    def setUp(self):
        # generate sources with independent variance modulations, the
        # first source will be the target source
        z = np.abs(np.random.randn(self.EPOCHS, self.SOURCES))
        for i in range(self.SOURCES):
            z[:, i] /= z[:, i].std()
        self.s = np.random.randn(self.EPOCHS, self.SAMPLES, self.SOURCES)
        for i in range(self.SOURCES):
            for j in range(self.EPOCHS):
                self.s[j, :, i] *= z[j, i]
        # the mixmatrix which converts our sources to channels
        # X = As + noise
        self.A = np.random.randn(self.CHANNELS, self.SOURCES)
        # our 'signal' which 50 epochs, 100 samples and 10 channels
        self.X = np.empty((self.EPOCHS, self.SAMPLES, self.CHANNELS))
        for i in range(self.EPOCHS):
            self.X[i] = np.dot(self.A, self.s[i].T).T
        noise = np.random.randn(self.EPOCHS, self.SAMPLES, self.CHANNELS) * 0.01
        self.X += noise
        # convert to epo
        axes = [z[:, 0], np.arange(self.X.shape[1]), np.arange(self.X.shape[2])]
        self.epo = Data(self.X,
                axes=axes,
                names=['target_variable', 'time', 'channel'],
                units=['#', 'ms', '#'])

    def test_d(self):
        """Test if the list of lambdas is reverse-sorted and the first one > 0."""
        W, A_est, d = calculate_spoc(self.epo)
        self.assertTrue(d[0] > 0)
        self.assertTrue(np.all(d == np.sort(d)[::-1]))

    def test_A(self):
        """Test if A_est is elementwise almost equal A."""
        W, A_est, d = calculate_spoc(self.epo)
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
        W, A_est, d = calculate_spoc(self.epo)
        # applying the filter to X gives us s_est which should be almost
        # equal s
        s_est = np.empty(self.s.shape)
        for i in range(self.EPOCHS):
            s_est[i] = np.dot(self.X[i], W[:, [0, -1]])
        # correct for scaling, and sign
        self.s = self.s.reshape(-1, self.SOURCES)
        s_est2 = s_est.reshape(-1, self.SOURCES)
        epsilon = 0.01

        idx = np.argmax(np.abs(s_est2[:, 0]))
        s_est2[:, 0] /= s_est2[idx, 0]
        idx = np.argmax(np.abs(self.s[:, 0]))
        self.s[:, 0] /= self.s[idx, 0]
        diff = np.sum(np.abs(self.s[:, 0] - s_est2[:, 0])) / self.s.shape[0]
        self.assertTrue(diff < epsilon)

    #def test_calculate_signed_r_square_swapaxes(self):
    #    """caluclate_r_square must work with nonstandard classaxis."""
    #    dat = calculate_signed_r_square(swapaxes(self.dat, 0, 2), classaxis=2)
    #    # the class-axis just dissapears during
    #    # calculate_signed_r_square, so axis 2 becomes axis 1
    #    dat = dat.swapaxes(0, 1)
    #    dat2 = calculate_signed_r_square(self.dat)
    #    np.testing.assert_array_equal(dat, dat2)

    def test_calculate_signed_r_square_copy(self):
        """caluclate_r_square must not modify argument."""
        cpy = self.epo.copy()
        calculate_spoc(self.epo)
        self.assertEqual(self.epo, cpy)


if __name__ == '__main__':
    unittest.main()
