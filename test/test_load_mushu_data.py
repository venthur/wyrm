#!/usr/bin/env python


import unittest
import os
import json
import struct

import numpy as np

from wyrm.io import load_mushu_data


class TestLoadMushuData(unittest.TestCase):

    def setUp(self):
        with open('foo.meta', 'w') as fh:
            json.dump({'Sampling Frequency' : 1,
                       'Channels' : ['ch1', 'ch2'],
                       'Amp': 'Dummy Amp'}, fh)
        with open('foo.marker', 'w') as fh:
            fh.write('0.0 Marker 0\n')
            fh.write('500.0 Marker 500\n')
            fh.write('666.666 Marker 666.666\n')
        with open('foo.eeg', 'wb') as fh:
            for i in (0, 1, 33, 66, .1, .2):
                fh.write(struct.pack('f', i))

    def tearDown(self):
        for fname in 'foo.meta', 'foo.eeg', 'foo.markers':
            try:
                os.remove(fname)
            except:
                pass

    def test_incomplete_fileset(self):
        """Must raise an error if not all files are available."""
        os.remove('foo.marker')
        with self.assertRaises(AssertionError):
            load_mushu_data('foo.meta')

    def test_loading(self):
        """Loads data correctly."""
        dat = load_mushu_data('foo.meta')
        self.assertEqual(dat.fs, 1)
        np.testing.assert_array_equal(dat.markers, [[0.0, 'Marker 0'], [500.0, 'Marker 500'], ['666.666', 'Marker 666.666']])
        np.testing.assert_array_equal(dat.axes[0], [0, 1000, 2000])
        np.testing.assert_array_equal(dat.axes[1], ['ch1', 'ch2'])
        self.assertEqual(dat.names, ['time', 'channels'])
        self.assertEqual(dat.units, ['ms', '#'])
        data = np.array([0, 1, 33, 66, .1, .2], np.float32).reshape(-1, 2)
        np.testing.assert_array_equal(dat.data, data)


if __name__ == '__main__':
    unittest.main()

