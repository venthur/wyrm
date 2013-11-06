#!/usr/bin/env python

"""Plotting methods.

This module contains various plotting methods.

"""

from __future__ import division

import numpy as np
import random as rnd
import inspect
from matplotlib import pyplot as plt
from wyrm.types import Data
#from scipy import interpolate

import tentensystem as tts

### The old plotting functions #########################################################

def plot_scalp(v, channel):
    """Plot the values v for channel ``channel`` on a scalp."""

    channelpos = [tts.channels[c] for c in channel]
    points = [calculate_stereographic_projection(i) for i in channelpos]
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    z = v
    X, Y, Z = interpolate_2d(x, y, z)
    plt.contour(X, Y, Z, 20)
    plt.contourf(X, Y, Z, 20)
    #plt.clabel(im)
    plt.colorbar()
    plt.gca().add_artist(plt.Circle((0, 0), radius=1, linewidth=3, fill=False))
    plt.plot(x, y, 'bo')
    for i in zip(channel, zip(x,y)):
        plt.annotate(i[0], i[1])


def plot_channels(dat, chanaxis=-1, otheraxis=-2):
    """Plot all channels for a continuous.

    Parameters
    ----------
    dat : Data

    """
    ax = []
    n_channels = dat.data.shape[chanaxis]
    for i, chan in enumerate(dat.axes[chanaxis]):
        if i == 0:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1)
        else:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1, sharex=ax[0], sharey=ax[0])
        ax.append(a)
        x, y =  dat.axes[otheraxis], dat.data.take([i], chanaxis)
        a.plot(dat.axes[otheraxis], dat.data.take([i], chanaxis).squeeze())
        a.set_title(chan)
        plt.axvline(x=0)
        plt.axhline(y=0)


def plot_spectrum(spectrum, freqs):
    plt.plot(freqs, spectrum, '.')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('[dl]')


def plot_spectrogram(spectrogram, freqs):
    extent = 0, len(spectrogram), freqs[0], freqs[-1]
    plt.imshow(spectrogram.transpose(),
        aspect='auto',
        origin='lower',
        extent=extent,
        interpolation='none')
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time')


def calculate_stereographic_projection(p):
    """Calculate the stereographic projection.

    Given a unit sphere with radius ``r = 1`` and center at the origin.
    Project the point ``p = (x, y, z)`` from the sphere's South pole (0,
    0, -1) on a plane on the sphere's North pole (0, 0, 1).

    The formula is:

        P' = P * (2r / (r + z))

    Parameters
    ----------
    p : [float, float]
        The point to be projected in cartesian coordinates.

    Returns
    -------
    x, y : float, float
        The projected point on the plane.

    """
    # P' = P * (2r / r + z)
    mu = 1 / (1 + p[2])
    x = p[0] * mu
    y = p[1] * mu
    return x, y


def interpolate_2d(x, y, z):
    """Interpolate missing points on a plane.

    Parameters
    ----------
    x, y, z : equally long lists of floats
        1d arrays defining points like ``p[x, y] = z``

    Returns
    -------
    X, Y, Z : 1d array, 1d array, 2d array
        ``Z`` is a 2d array ``[min(x)..max(x), [min(y)..max(y)]`` with
        the interpolated values as values.

    """
    X = np.linspace(min(x), max(x))
    Y = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(X, Y)
    #f = interpolate.interp2d(x, y, z)
    #Z = f(X[0, :], Y[:, 0])
    f = interpolate.LinearNDInterpolator(zip(x, y), z)
    Z = f(X, Y)
    return X, Y, Z

### The new plotting functions #############################################

# automated creation of more realistic test data ##########
# test function: (x^3 * cos(x)) / 20, x in [-5, -2] (with variations)

figures = []
plots = []

def create_data(channel_count = 2, steps = 100):

    def create_channel(steps = 100):
        steps = float(steps)
        a = -5
        b = -2
        rnd_fac1 = rnd.randrange(25, 200) / 100.
        rnd_fac2 = rnd.randrange(-200, 200) / 100.
        rnd_fac3 = rnd.randrange(-100, 100) / 100.
        range_x = np.arange(a, b, np.absolute(a-b)/steps)
        range_y = np.zeros(steps)
        
        cnt = 0
        for i in range_x:
            range_y[cnt] = (i**3 * np.cos(i - rnd_fac3) / 20) * rnd_fac1 - rnd_fac2
            cnt += 1
        
        return range_y
    
    data = np.zeros([steps, channel_count])
    channels = []
    for i in range(channel_count):
        data[:,i] = create_channel(steps)
        channels.append('ch' + str(i))
        
    axes = [np.arange(0,steps*10, 10), channels]
    names = ["time", "channel"]
    units = ["ms", "stuff"]
    dat = Data(data, axes, names, units)
    return (dat)


# plots a simple time_interval with the given data
def plot_timeinterval(data, highlights=None, legend=True, show=True, save=False, save_name='timeinterval', save_path=None):

    plt.clf()

    # plotting of the data
    plt.plot(data.axes[0], data.data)
    
    # filling of global lists for figures and axes (plots) 
    global plots, figures
    figures = [plt.gcf()]
    plots = [plt.gca()]
    
    # plotting of highlights
    add_highlights(highlights)

    # labeling of axes
    plt.xlabel(data.units[0])
    plt.ylabel("$\mu$V", rotation = 0)
    
    # labeling of channels
    if legend: plt.legend(data.axes[1])
    
    # saving if specified
    if save:
        if save_path is None:
            plt.savefig(save_name, bbox_inches='tight')
        else:
            plt.savefig(save_path + save_name, bbox_inches='tight')
        
    plt.grid(True)
    if show: plt.show()


# Adds highlights to the specified axes.
# axes: a list of axes
# obj_highlight: an instance of the Highlight class
def add_highlights(obj_highlight, axes = None):
    
    global plots
    if axes is None:
        axes = plots
    
    def highlight(start, end, axis, color, alpha):
        axis.axvspan(start, end, edgecolor='w', facecolor=color, alpha=alpha)
        # the edges of the box are at the moment white. transparent edges would be better.
    
    # check if obj_highlight is an instance of the Highlight class
    if isinstance(obj_highlight, type(Highlight())):
        #print('instance found')
        for p in axes:
            for hl in obj_highlight.spans:
                highlight(hl[0], hl[1], axis = p, color=obj_highlight.color, alpha=obj_highlight.alpha)

# class for highlights.
# spans: list of two-element lists "[[start1, end1], ..., [startn, endn]]"
# color: color of the highlighted area
# alpha: transparency of the highlighted area
class Highlight:

    def __init__(self, spans=[], color='#b3b3b3', alpha=0.5):
        for hl in spans:
            if len(hl) != 2:
                print("'spans' has wrong form. Usage: [[start1, end1], ..., [startn, endn]].")
                self.spans = None
                break
        else:
            self.spans = spans
        self.color = color
        self.alpha = alpha

    def toString(self):
        s = ['spans: ' + str(self.spans), 'color: ' + str(self.color), 'alpha: ' + str(self.alpha)]
        print(', '.join(s))
