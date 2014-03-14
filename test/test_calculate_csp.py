from __future__ import division

import unittest

import numpy as np
np.random.seed(42)

from wyrm.types import Data
from wyrm.processing import calculate_csp


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

        a = np.array([1 for i in range(self.X.shape[0])])
        a[0::2] = 0
        axes = [a, np.arange(self.X.shape[1]), np.arange(self.X.shape[2])]
        self.epo = Data(self.X, axes=axes, names=['class', 'time', 'channel'], units=['#', 'ms', '#'])
        self.epo.class_names = ['foo', 'bar']

    def test_d(self):
        """Test if the first lambda is almost 1 and the last one almost -1."""
        W, A_est, d = calculate_csp(self.epo)
        epsilon = 0.1
        self.assertAlmostEqual(d[0], 1, delta=epsilon)
        self.assertAlmostEqual(d[-1], -1, delta=epsilon)

    def test_A(self):
        """Test if A_est is elementwise almost equal A."""
        W, A_est, d = calculate_csp(self.epo)
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
        W, A_est, d = calculate_csp(self.epo)
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

    def test_manual_class_selection(self):
        """Manual class indices selection must work."""
        w, a, d = calculate_csp(self.epo)
        w2, a2, d2 = calculate_csp(self.epo, [0, 1])
        np.testing.assert_array_equal(w, w2)
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(d, d2)
        w2, a2, d2 = calculate_csp(self.epo, [1, 0])
        np.testing.assert_array_almost_equal(np.abs(w), np.abs(w2[:, ::-1]))
        np.testing.assert_array_almost_equal(np.abs(a), np.abs(a2[:, ::-1]))
        np.testing.assert_array_almost_equal(np.abs(d), np.abs(d2[::-1]))

    def test_raise_error_on_wrong_manual_classes(self):
        """Raise error if classes not in epo."""
        with self.assertRaises(AssertionError):
            calculate_csp(self.epo, [0, 2])
            calculate_csp(self.epo, [0, -1])

    def test_raise_error_with_automatic_classes(self):
        """Raise error if not enough classes in epo."""
        self.epo.axes[0][:] = 0
        with self.assertRaises(AssertionError):
            calculate_csp(self.epo)


    #def test_calculate_csp_swapaxes(self):
    #    """caluclate_csp must work with nonstandard classaxis."""
    #    dat = calculate_csp(swapaxes(self.epo, 0, 2), classaxis=2, chanaxis=0)
    #    dat2 = calculate_csp(self.epo)
    #    np.testing.assert_array_equal(dat[0], dat2[0])

    def test_calculate_csp_copy(self):
        """caluclate_csp must not modify argument."""
        cpy = self.epo.copy()
        calculate_csp(self.epo)
        self.assertEqual(self.epo, cpy)


if __name__ == '__main__':
    unittest.main()
