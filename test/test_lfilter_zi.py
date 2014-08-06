from __future__ import division

import unittest

import numpy as np

from wyrm.processing import lfilter_zi


class TestLFilterZi(unittest.TestCase):

    COEFFS = 10

    def setUp(self):
        self.b, self.a = np.ones(self.COEFFS), np.ones(self.COEFFS)

    def test_lfilter_1d(self):
        """Output has the correct shape for n=1."""
        zi = lfilter_zi(self.b, self.a)
        self.assertEqual(len(zi), self.COEFFS - 1)

    def test_lfilter_nd(self):
        """Output has the correct shape for n>1."""
        zi = lfilter_zi(self.b, self.a, 7)
        self.assertEqual(zi.shape, (self.COEFFS - 1, 7))


if __name__ == '__main__':
    unittest.main()
