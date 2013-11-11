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
def create_data_ti(channel_count = 2, steps = 100):
    
    data = np.zeros([steps, channel_count])
    channels = []
    for i in range(channel_count):
        data[:,i] = _create_channel(steps)
        channels.append('ch' + str(i))
        
    axes = [np.arange(0,steps*10, 10), channels]
    names = ["time", "channel"]
    units = ["ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return (dat)

def create_epoched_data_ti(class_count = 4, channel_count = 2, steps = 100):
    data = np.zeros([class_count, steps, channel_count])
    for i in range(class_count):
        for o in range(channel_count):
            data[i,:,o] = _create_channel(steps)
            
    # create the channel labels
    channels = []
    for i in range(channel_count):
        channels.append('ch' + str(i))
        
    # create the class labels
    classes = []
    for i in range(class_count):
        classes.append('class' + str(i%2))
        
    axes = [classes, np.arange(0,steps*10, 10), channels]
    names = ["class", "time", "channel"]
    units = ["class_stuff", "ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return(dat)

def _create_channel(steps = 100):
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

# plots a simple time_interval with the given data
def plot_timeinterval(data, highlights=None, legend=True, show=True, save=False, save_name='timeinterval', save_path=None):

    plt.clf()

    # plotting of the data
    plt.plot(data.axes[0], data.data)
    
    # plotting of highlights
    add_highlights(highlights)

    # labeling of axes
    add_labels(data.units[0], "$\mu$V")
    #plt.xlabel(data.units[0])
    #plt.ylabel("$\mu$V", rotation = 0)
    
    # labeling of channels
    if legend: plt.legend(data.axes[1])
    
    # saving if specified
    if save:
        if save_path is None:
            plt.savefig(save_name, bbox_inches='tight')
        else:
            plt.savefig(save_path + save_name, bbox_inches='tight')
        
    plt.grid(True)
    
    # showing if specified
    if show: plt.show()

# adds a subplot to the current figure at the specified position.
# data: wyrm Data
# position: position of the subplot
# epoch: specifies the epoch to plot
# highlights (optional): a wyrm.plot.Highlight object to create highlights
# legend (optional): boolean to switch the legend on or off 
def _subplot_timeinterval(data, position, epoch, highlights=None, legend=True):
    
    # plotting of the data
    plt.subplot(position)
    plt.plot(data.axes[len(data.axes) - 2], data.data[epoch])
    
    # plotting of highlights
    add_highlights(highlights, axes=[plt.gca()])

    # labeling of axes
    plt.xlabel(data.units[len(data.axes) - 2])
    plt.ylabel("$\mu$V", rotation = 0)
    
    # labeling of channels
    if legend: plt.legend(data.axes[len(data.axes) - 1])
    
    plt.grid(True)
    
def plot_epoched_timeinterval(data, highlights=None, legend=True, show=True, save=False, save_name='epoched_timeinterval', save_path=None):
    
    plt.clf()
    
    # iterate over epochs
    for i in range(len(data.data)):
        pos = int('1' + str(len(data.data)) + str(i+1))
        _subplot_timeinterval(data, pos, i, highlights, legend)
        
    # adjust the spacing
    plt.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.04, hspace=0.3, wspace=0.3)
    
    # saving if specified
    if save:
        if save_path is None:
            plt.savefig(save_name, bbox_inches='tight')
        else:
            plt.savefig(save_path + save_name, bbox_inches='tight')
    
    # showing if specified
    if show: plt.show()

# Adds highlights to the specified axes.
# obj_highlight: an instance of the Highlight class
# axes (optional): a list of axes
def add_highlights(obj_highlight, axes=None):
    
    if axes is None:
        axes = plt.gcf().axes
    
    def highlight(start, end, axis, color, alpha):
        axis.axvspan(start, end, edgecolor='w', facecolor=color, alpha=alpha)
        # the edges of the box are at the moment white. transparent edges would be better.
    
    # check if obj_highlight is an instance of the Highlight class
    if isinstance(obj_highlight, type(Highlight())):
        for p in axes:
            for hl in obj_highlight.spans:
                highlight(hl[0], hl[1], p, obj_highlight.color, obj_highlight.alpha)
                
# Adds labels to the specified axes (default: all axes of current figure)
# xLabels: String to label the x-axis
# yLabels: String to label the y-axis
# axes (optional): List of matplotlib.Axes to apply the labels on 
def add_labels(xLabel, yLabel, axes=None):
    
    if axes is None:
        axes = plt.gcf().axes
        
    # labeling of axes
    for ax in axes:
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel, rotation = 0)

# class for highlights.
# spans: list of two-element lists "[[start1, end1], ..., [startn, endn]]"
# color (optional): color of the highlighted area
# alpha (optional): transparency of the highlighted area
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
