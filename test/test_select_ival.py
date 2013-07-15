

import unittest

import numpy as np

from wyrm.misc import Epo
from wyrm.misc import select_ival


class TestSelectIval(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        fs = 10
        marker = []
        ival = [-1000, 1000]
        classes = [0, 0, 0]
        class_names = ['class 1']
        # three cnts: 1s, -1s, and 0s
        data = np.array([ones, ones * -1, ones * 0])
        self.epo = Epo(data, fs, channels, marker, classes, class_names, ival)

    def test_select_ival(self):
        """Selecting Intervals."""
        # normal case
        epo = select_ival(self.epo, [-500, 0])
        self.assertEqual(epo.ival, [-500, 0])
        # the full epo.ival
        epo = select_ival(self.epo, self.epo.ival)
        self.assertEqual(epo.ival, self.epo.ival)
        np.testing.assert_array_equal(epo.data, self.epo.data)

    def test_ival_checks(self):
        """Test for malformed ival parameter."""
        with self.assertRaises(AssertionError):
            select_ival(self.epo, [0, -1])
        with self.assertRaises(AssertionError):
            select_ival(self.epo, [self.epo.ival[0]-1, 0])
        with self.assertRaises(AssertionError):
            select_ival(self.epo, [0, self.epo.ival[1]+1])



if __name__ == '__main__':
    unittest.main()
