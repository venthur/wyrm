#!/usr/bin/env python


import unittest

import numpy as np

from wyrm.plot import _interpolate_2d


class TestInterpolate2d(unittest.TestCase):

    def test_max(self):
        x = [1, 2, 1, 2, 3]
        y = [1, 1, 2, 2, 3]
        z = [10, 10, 10, 10, 2]
        _, _, Z = _interpolate_2d(x, y, z)
        self.assertAlmostEqual(np.nanmax(Z), 10.)


if __name__ == '__main__':
    unittest.main()
