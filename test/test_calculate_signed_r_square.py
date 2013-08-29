from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import calculate_signed_r_square
from wyrm.processing import swapaxes


class TestSelectClasses(unittest.TestCase):

    def setUp(self):
        # create noisy data [40 epochs, 100 samples, 64 channels] with
        # values 0..1
        dat = np.random.uniform(size=(40, 100, 64))
        # every second epoch belongs to class 0 and 1 alterning
        # for class 1 add 1 in interval 40..80
        # for class 2 add 1 in interval 20..60
        #
        #        * .* .
        #
        # .* .* .      * .*
        # ------------------>
        # 0  20 40 60 80 100
        dat[::2,40:80:,:] += 1
        dat[1::2,20:60:,:] += 1
        time = np.arange(dat.shape[1])
        classes = np.zeros(dat.shape[0])
        classes[::2] = 1
        chans = np.arange(64)
        self.dat = Data(dat, [classes, time, chans], ['class', 'time', 'channel'], ['#', 'ms', '#'])
        self.dat.class_names = 'one', 'two'

    def test_calculate_signed_r_square(self):
        """Calculating signed r**2."""
        dat = calculate_signed_r_square(self.dat)
        self.assertEqual(dat.ndim + 1, self.dat.data.ndim)
        # average over channels (one could also take just one channel)
        dat = dat.mean(axis=1)
        # check the intervals
        self.assertTrue(all(dat[0:20] < .2))
        self.assertTrue(all(dat[20:40] > .5))
        self.assertTrue(all(dat[40:60] < .2))
        self.assertTrue(all(dat[60:80] < .5))
        self.assertTrue(all(dat[80:100] < .2))

    def test_calculate_signed_r_square_min_max(self):
        """Min and max values must be in [-1, 1]."""
        dat = calculate_signed_r_square(self.dat)
        self.assertTrue(-1 <= np.min(dat) <= 1)
        self.assertTrue(-1 <= np.max(dat) <= 1)

    def test_calculate_signed_r_square_with_cnt(self):
        """Select epochs must raise an exception if called with cnt argument."""
        del(self.dat.class_names)
        with self.assertRaises(AssertionError):
            calculate_signed_r_square(self.dat)

    def test_calculate_signed_r_square_swapaxes(self):
        """caluclate_r_square must work with nonstandard classaxis."""
        dat = calculate_signed_r_square(swapaxes(self.dat, 0, 2), classaxis=2)
        # the class-axis just dissapears during
        # calculate_signed_r_square, so axis 2 becomes axis 1
        dat = dat.swapaxes(0, 1)
        dat2 = calculate_signed_r_square(self.dat)
        np.testing.assert_array_equal(dat, dat2)

    def test_calculate_signed_r_square_copy(self):
        """caluclate_r_square must not modify argument."""
        cpy = self.dat.copy()
        calculate_signed_r_square(self.dat)
        self.assertEqual(self.dat, cpy)


if __name__ == '__main__':
    unittest.main()
