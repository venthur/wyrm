from __future__ import division

import unittest

import numpy as np

from wyrm.plot import interpolate_2d


class TestInterpolate2d(unittest.TestCase):

    def test_max(self):
        """Make sure the interpolation does not lead to interpolated values
        bigger than the maximum provided by the points.

        This particularly happens when doing interpolation with splines.

        """
        x = [1, 2, 1, 2, 3]
        y = [1, 1, 2, 2, 3]
        z = [10, 10, 10, 10, 2]
        _, _, Z = interpolate_2d(x, y, z)
        self.assertAlmostEqual(np.nanmax(Z), 10.)


if __name__ == '__main__':
    unittest.main()
