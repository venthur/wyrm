from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import create_feature_vectors
from wyrm.processing import swapaxes

class TestCreateFeatureVectors(unittest.TestCase):

    def setUp(self):
        # create epoched data with only 0s in class0, 1s in class1 and
        # 2s in class2
        cnt = np.ones((10, 3))
        epo = np.array([0*cnt, 1*cnt, 2*cnt])
        time = np.arange(10)
        channels = np.array(['ch1', 'ch2', 'ch3'])
        classes = np.arange(3)
        axes = ['class', 'time', 'channel']
        units = ['#', 'ms', '#']
        self.dat = Data(epo, [classes, time, channels], axes, units)

    def test_create_feature_vectors(self):
        """Create Feature Vectors."""
        dat = create_feature_vectors(self.dat)
        self.assertTrue(all(dat.data[0] == 0))
        self.assertTrue(all(dat.data[1] == 1))
        self.assertTrue(all(dat.data[2] == 2))
        self.assertEqual(dat.data.ndim, 2)
        self.assertEqual(len(dat.axes), 2)
        self.assertEqual(len(dat.names), 2)
        self.assertEqual(len(dat.units), 2)
        self.assertEqual(dat.names[-1], 'feature vector')
        self.assertEqual(dat.units[-1], 'dl')
        np.testing.assert_array_equal(dat.axes[-1], np.arange(dat.data.shape[-1]))

    def test_create_feature_vectors_copy(self):
        """create_feature_vectors must not modify argument."""
        cpy = self.dat.copy()
        create_feature_vectors(self.dat)
        self.assertEqual(cpy, self.dat)

    def test_create_feature_vectors_swapaxes(self):
        """create_feature_vectors must work with nonstandard classaxis."""
        # keep in mind that create_feature_vectors already swaps the
        # axes internally to move the classaxis to 0
        dat = create_feature_vectors(swapaxes(self.dat, 0, 2), classaxis=2)
        dat2 = create_feature_vectors(self.dat)
        self.assertEqual(dat, dat2)


if __name__ == '__main__':
    unittest.main()

