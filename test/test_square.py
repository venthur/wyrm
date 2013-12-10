from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import square

class TestSquare(unittest.TestCase):

    def setUp(self):
        raw = np.arange(1, 21).reshape(4, 5)
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        time = np.arange(4)
        self.dat = Data(raw, [time, channels], ['time', 'channels'], ['ms', '#'])

    def test_square(self):
        """Square basics must work."""
        dat = square(self.dat)
        # works elementwise (does not alter the shape)
        self.assertEqual(self.dat.data.shape, dat.data.shape)
        # actual square was computed
        np.testing.assert_array_almost_equal(dat.data, np.square(self.dat.data))

    def test_square_copy(self):
        """Square must not change the original parameter."""
        cpy = self.dat.copy()
        square(self.dat)
        self.assertEqual(cpy, self.dat)

if __name__ == '__main__':
    unittest.main()
