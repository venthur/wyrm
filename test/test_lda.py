#!/usr/bin/env python


import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import lda_train, lda_apply


class TestLDA(unittest.TestCase):

    def test_correct_classlabels(self):
        """lda_train must throw an error if the class labels are not exactly [0, 1]."""
        data = np.random.random((50, 100))
        labels = np.zeros(50)
        # only 0s -> fail
        fv = Data(data=data, axes=[labels, np.arange(100)], units=['x', 'y'], names=['foo', 'bar'])
        with self.assertRaises(ValueError):
            lda_train(fv)
        # 0s and 1s -> ok
        labels[1] = 1
        fv = Data(data=data, axes=[labels, np.arange(100)], units=['x', 'y'], names=['foo', 'bar'])
        try:
            lda_train(fv)
        except ValueError:
            self.fail()
        # 0s, 1s, and 2s -> fail
        labels[2] = 2
        fv = Data(data=data, axes=[labels, np.arange(100)], units=['x', 'y'], names=['foo', 'bar'])
        with self.assertRaises(ValueError):
            lda_train(fv)


    def test_lda_apply(self):
        """trivial lda application must work."""
        # this is not a proper test for LDA
        data = np.random.random((50, 100))
        labels = np.zeros(50)
        data[::2] += 1
        labels[::2] += 1
        fv = Data(data=data, axes=[labels, np.arange(100)], units=['x', 'y'], names=['foo', 'bar'])
        clf = lda_train(fv)
        out = lda_apply(fv, clf)
        # map projections back to 0s and 1s
        out[out > 0] = 1
        out[out < 0] = 0
        np.testing.assert_array_equal(out, labels)


    def test_lda_apply_w_shrinkage(self):
        """trivial lda application must work."""
        # this is not a proper test for LDA
        data = np.random.random((50, 100))
        labels = np.zeros(50)
        data[::2] += 1
        labels[::2] += 1
        fv = Data(data=data, axes=[labels, np.arange(100)], units=['x', 'y'], names=['foo', 'bar'])
        clf = lda_train(fv, shrink=True)
        out = lda_apply(fv, clf)
        # map projections back to 0s and 1s
        out[out > 0] = 1
        out[out < 0] = 0
        np.testing.assert_array_equal(out, labels)


if __name__ == '__main__':
    unittest.main()
