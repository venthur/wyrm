#!/usr/bin/env python


import unittest

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from wyrm.types import Data
from wyrm.tentensystem import channels
from wyrm import plot


class TestPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = np.linspace(0, 1000, 1000, endpoint=False)
        y = 5 * np.sin(2 * np.pi * x) + np.random.normal(0, 1, 1000)
        data = np.array([y]*len(channels)).T
        axes = [x, channels.keys()]
        cls.cnt = Data(data=data, axes=axes, names=['time', 'channel'], units=['ms', '#'])

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
        plt.suptitle('plot_channels')

    def test_plot_spatio_temporal_r2_values(self):
        plot.plot_spatio_temporal_r2_values(self.cnt)

    def test_plot_spectrum(self):
        plot.plot_spectrum(np.random.random(100), np.arange(100))

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
