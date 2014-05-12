#!/usr/bin/env python


import unittest

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from wyrm.types import Data
from wyrm import plot
from wyrm.processing import CHANNEL_10_20


class TestPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = np.linspace(0, 1000, 1000, endpoint=False)
        y = 5 * np.sin(2 * np.pi * x)
        data = np.tile(y[:, None], (1, len(CHANNEL_10_20)))
        data += np.random.normal(size=data.shape)
        axes = [x, [i[0] for i in CHANNEL_10_20]]
        cls.cnt = Data(data=data, axes=axes, names=['time', 'channel'], units=['ms', '#'])
        cls.cnt.fs = 1000

        classes = [0, 1] * 5
        data = np.array([data]*10)
        data[::2] *= 0.5
        axes = [classes, x, [i[0] for i in CHANNEL_10_20]]
        cls.epo = Data(data=data, axes=axes, names=['class', 'time', 'channel'], units=['#', 'ms', '#'])
        cls.epo.fs = 1000
        cls.epo.class_names = ['class 1', 'class 2']

        plot.beautify()

    def setUp(self):
        plt.figure(figsize=(16, 9))

    def tearDown(self):
        test_method = self._testMethodName
        plt.suptitle(test_method)
        plt.savefig(test_method + '.png')
        plt.close()

###############################################################################

    def test_plot_channels(self):
        plot.plot_channels(self.cnt)

    def test_plot_spatio_temporal_r2_values(self):
        plot.plot_spatio_temporal_r2_values(self.epo)

    def test_plot_spectrogram(self):
        plot.plot_spectrogram(np.random.random((10, 100)), np.arange(100))

###############################################################################

    def test_plot_timeinterval(self):
        _, ax = plt.subplots(2, 1)
        plt.sca(ax[0])
        plot.plot_timeinterval(self.cnt, np.random.random(1000), [[200, 300], [500, 700]])
        plt.sca(ax[1])
        plot.plot_timeinterval(self.cnt)

    def test_plot_tenten(self):
        plot.plot_tenten(self.cnt)

    def test_plot_scalp(self):
        plot.plot_scalp(self.cnt.data[0, :], self.cnt.axes[-1])

    def test_plot_scalp_ti(self):
        plot.plot_scalp_ti(self.cnt.data[0, :], self.cnt.axes[-1], self.cnt, [100, 700])


if __name__ == '__main__':
    unittest.main()
