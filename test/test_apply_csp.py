from __future__ import division

import unittest

import numpy as np
np.random.seed(42)

from wyrm.types import Data
from wyrm.processing import apply_csp


class TestApplyCSP(unittest.TestCase):

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

        self.filter = np.random.random((self.CHANNELS, self.CHANNELS))

    def test_apply_csp(self):
        """apply_csp."""
        dat = apply_csp(self.epo, self.filter)
        # reduce the channels down to 2, the rest of the shape should
        # stay the same
        self.assertEqual(self.epo.data.shape[0], dat.data.shape[0])
        self.assertEqual(self.epo.data.shape[1], dat.data.shape[1])
        self.assertEqual(2, dat.data.shape[2])
        # new name for csp axis
        self.assertEqual(dat.names[-1], 'CSP Channel')
        # check if the dot product was calculated correctly
        d = np.array([np.dot(self.epo.data[i], self.filter[:, [0, -1]]) for i in range(self.epo.data.shape[0])])
        np.testing.assert_array_equal(d, dat.data)


    #def test_apply_csp_swapaxes(self):
    #    """apply_csp must work with nonstandard classaxis."""
    #    dat = apply_csp(swapaxes(self.epo, 0, 1), self.filter.T, classaxis=1)
    #    #dat = swapaxes(dat, 0, 1)
    #    print dat.data.shape
    #    dat2 = apply_csp(self.epo, self.filter)
    #    print
    #    print dat2.data.shape
    #    self.assertEqual(dat, dat2)

    def test_apply_csp_copy(self):
        """apply_csp must not modify argument."""
        cpy = self.epo.copy()
        apply_csp(self.epo, self.filter)
        self.assertEqual(self.epo, cpy)


if __name__ == '__main__':
    unittest.main()
