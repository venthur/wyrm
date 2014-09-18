#!/usr/bin/env python


import unittest
import os

import numpy as np

from wyrm.types import Data
from wyrm.io import load, save


FILENAME = 'test_load_save.npy'


class TestLoadMushuData(unittest.TestCase):

    def setUp(self):
        data = np.arange(10).reshape(5, 2)
        axes = [np.arange(5), np.array(['ch1', 'ch2'])]
        names = ['time', 'channel']
        units = ['ms', '#']
        fs = 1000
        self.dat = Data(data, axes, names, units)
        self.dat.fs = fs

    def tearDown(self):
        try:
            os.remove(FILENAME)
        except:
            pass

    def test_load_save(self):
        save(self.dat, FILENAME)
        dat = load(FILENAME)
        self.assertTrue(isinstance(dat, Data))
        self.assertEqual(self.dat, dat)


if __name__ == '__main__':
    unittest.main()

