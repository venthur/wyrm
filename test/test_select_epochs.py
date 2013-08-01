


import unittest

import numpy as np

from wyrm.misc import Epo
from wyrm.misc import select_epochs


class TestSelectEpochs(unittest.TestCase):

    def setUp(self):
        ones = np.ones((10, 5))
        channels = ['ca1', 'ca2', 'cb1', 'cb2', 'cc1']
        fs = 10
        marker = []
        t_start = -1000
        classes = [0, 1, 2, 1]
        class_names = ['zeros', 'ones', 'twoes']
        # three cnts: 1s, -1s, and 0s
        data = np.array([ones * 0, ones * 1, ones * 2, ones * 0])
        self.epo = Epo(data, fs, channels, marker, classes, class_names, t_start)

    def test_select_epochs(self):
        """Selecting Epochs."""
        # normal case
        epo = select_epochs(self.epo, [0])
        self.assertEqual(epo.data.shape[0], 1)
        np.testing.assert_array_equal(epo.data, self.epo.data[[0]])
        np.testing.assert_array_equal(epo.classes, self.epo.classes[[0]])
        # normal every second
        epo = select_epochs(self.epo, [0, 2])
        self.assertEqual(epo.data.shape[0], 2)
        np.testing.assert_array_equal(epo.data, self.epo.data[::2])
        np.testing.assert_array_equal(epo.classes, self.epo.classes[::2])
        # the full epo
        epo = select_epochs(self.epo, range(self.epo.data.shape[0]))
        np.testing.assert_array_equal(epo.data, self.epo.data)
        np.testing.assert_array_equal(epo.classes, self.epo.classes)
        # remove one
        epo = select_epochs(self.epo, [0], invert=True)
        self.assertEqual(epo.data.shape[0], 3)
        np.testing.assert_array_equal(epo.data, self.epo.data[1:])
        np.testing.assert_array_equal(epo.classes, self.epo.classes[1:])
        # remove every second
        epo = select_epochs(self.epo, [0, 2], invert=True)
        self.assertEqual(epo.data.shape[0], 2)
        np.testing.assert_array_equal(epo.data, self.epo.data[1::2])
        np.testing.assert_array_equal(epo.classes, self.epo.classes[1::2])

if __name__ == '__main__':
    unittest.main()
