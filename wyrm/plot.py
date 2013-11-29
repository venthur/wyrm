#!/usr/bin/env python

"""Plotting methods.

This module contains various plotting methods.

"""

from __future__ import division

import numpy as np
import random as rnd
import inspect
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
import matplotlib.patches as patches
from wyrm.types import Data
from scipy import interpolate
import tentensystem as tts

### The old plotting functions ############################################################################################33

# def plot_scalp(v, channel):
#     """Plot the values v for channel ``channel`` on a scalp."""
# 
#     channelpos = [tts.channels[c] for c in channel]
#     points = [calculate_stereographic_projection(i) for i in channelpos]
#     x = [i[0] for i in points]
#     y = [i[1] for i in points]
#     z = v
#     X, Y, Z = interpolate_2d(x, y, z)
#     plt.contour(X, Y, Z, 20)
#     plt.contourf(X, Y, Z, 20)
#     #plt.clabel(im)
#     plt.colorbar()
#     plt.gca().add_artist(plt.Circle((0, 0), radius=1, linewidth=3, fill=False))
#     plt.plot(x, y, 'bo')
#     for i in zip(channel, zip(x,y)):
#         plt.annotate(i[0], i[1])


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

### The new plotting functions ###########################################################################################

# automated creation of more realistic test data #######################################################
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

def create_data_all():
    d = create_data_ti(channel_count=141)
    d.axes[1] = np.array(['Fp1','AFp1','Fpz','AFp2','Fp2','AF7','AF5','AF3','AFz','AF4','AF6','AF8','FAF5','FAF1','FAF2','FAF6',
'F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FFC9','FFC7','FFC5','FFC3','FFC1','FFC2',
'FFC4','FFC6','FFC8','FFC10','FT9','FT7','FC5','FC1','FCz','FC2','FC4','FC6','FT8','FT10','CFC9',
'CFC7','CFC5','CFC3','CFC1','CFC2','CFC4','CFC6','CFC8','CFC10','T9','T7','C5','C3','C1','Cz','C2',
'C4','C6','T8','T10','A1','CCP7','CCP5','CCP3','CCP1','CCP2','CCP4','CCP6','CCP8','A2','TP9','TP7','CP5',
'CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','PCP9','PCP7','PCP5','PCP3','PCP1','PCP2','PCP4','PCP6',
'PCP8','PCP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO9','PPO7','PPO5','PPO3','PPO1',
'PPO2','PPO4','PPO6','PPO8','PO10','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','O9','O1',
'OPO1','OPO2','O2','O10','OI2','Oz','OI1','I1','Iz','I2'])
    return d


def create_epoched_data_ti(epoch_count=4, channel_count=2, steps=100):
    data = np.zeros([epoch_count, steps, channel_count])
    for i in range(epoch_count):
        for o in range(channel_count):
            data[i,:,o] = _create_channel(steps)
            
    # create the channel labels
    channels = []
    for i in range(channel_count):
        channels.append('ch' + str(i))
        
    # create the class labels
    classes = []
    for i in range(epoch_count):
        classes.append('class' + str(i%2))
        
    axes = [classes, np.arange(0,steps*10, 10), channels]
    names = ["class", "time", "channel"]
    units = ["class_stuff", "ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return(dat)

def create_tenten_data(channel_count=21, steps=100):
    data = np.zeros([steps, channel_count])
    channels = []
    chan_names = ['A1', 'A2', 'C3', 'C4', 'Cz', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
         
    for i in range(channel_count):
        data[:,i] = _create_channel(steps)
        
    if channel_count <= len(chan_names):
        channels = chan_names[:channel_count]
    else:
        channels = chan_names
        i = len(chan_names)
        while (i < channel_count):
            channels.append('ch' + str(i))
            i = i+1

    axes = [np.arange(0,steps*10, 10), channels]
    names = ["time", "channel"]
    units = ["ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return (dat)

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

# \automated creation of more realistic test data ######################################################


# plots a simple time_interval with the given data
# data: wyrm.types.Data object containing the data to plot
# highlights (optional): a wyrm.plot.Highlight object to create highlights
# legend (optional): boolean to switch the legend on or off
# show (optional): boolean to switch immediate showing after creation
# save (optional): boolean to switch saving the created figure
# save_name (optional): String to specify the name the figure is saved as
# save_path (optional): String to specify the path the figure is saved to (usage: '/path/')  
# channel (optional): used for plotting only one specific channel
def plot_timeinterval(data, highlights=None, legend=True, show=True, save=False, save_name='timeinterval', save_path=None, channel=None):

    plt.clf()

    # plotting of the data
    if channel is None:
        plt.plot(data.axes[0], data.data)
    else:
        plt.plot(data.axes[0], data.data[:, channel])
    
    # plotting of highlights
    set_highlights(highlights)

    # labeling of axes
    set_labels(data.units[0], "$\mu$V", draw=False)
    
    # labeling of channels
    if legend:
        if channel is None:
            plt.legend(data.axes[1])
        else:
            plt.legend([data.axes[1][channel]])
    
    # saving if specified
    if save:
        if save_path is None:
            plt.savefig(save_name, bbox_inches='tight')
        else:
            plt.savefig(save_path + save_name, bbox_inches='tight')
        
    plt.grid(True)
    
    # showing if specified
    if show: plt.show()
    
    
# plots a series of time_intervals with the given epoched data
# data: wyrm.types.Data object containing the epoched data to plot
# highlights (optional): a wyrm.plot.Highlight object to create highlights
# legend (optional): boolean to switch the legend on or off
# show (optional): boolean to switch immediate showing after creation
# save (optional): boolean to switch saving the created figure
# save_name (optional): String to specify the name the figure is saved as
# save_path (optional): String to specify the path the figure is saved to (usage: '/path/') 
def plot_epoched_timeinterval(data, highlights=None, legend=True, show=True, save=False, save_name='epoched_timeinterval', save_path=None):
    
    plt.clf()
    
    # check of data is epoched
    if(len(data.data.shape) > 2):
        # iterate over epochs
        for i in range(len(data.data)):
            pos = int('1' + str(len(data.data)) + str(i+1))
            _subplot_timeinterval(data, pos, i, highlights, legend)
    else:
        pos = 111
        _subplot_timeinterval(data, pos, -1, highlights, legend)
        
    # add labels
    set_labels(data.units[len(data.axes) - 2], "$\mu$V", draw=False)
        
    # adjust the spacing
    plt.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.1, hspace=0.3, wspace=0.3)
    
    # saving if specified
    if save:
        if save_path is None:
            plt.savefig(save_name, bbox_inches='tight')
        else:
            plt.savefig(save_path + save_name, bbox_inches='tight')
    
    # showing if specified
    if show: plt.show()
    
    
# plots all recognized channels of the system according to their position on the scalp in a grid.
def plot_tenten(data, highlights=None, legend=True, show=True, save=False, save_name='system_plot', save_path=None):

    # this dictionary determines which y-position corresponds with which row in the grid
    ordering = {4.0  : 0,
                3.5  : 0,
                3.0  : 1,
                2.5  : 2,
                2.0  : 3,
                1.5  : 4,
                1.0  : 5,
                0.5  : 6,
                0.0  : 7,
                -0.5 : 8,
                -1.0 : 9,
                -1.5 : 10,
                -2.0 : 11,
                -2.5 : 12,
                -2.6 : 12,
                -3.0 : 13,
                -3.5 : 14,
                -4.0 : 15,
                -4.5 : 15,
                -5.0 : 16}
    
    # all the channels with their x- and y-position
    system = {
        'Fpz' : (0.0, 4.0),
        'Fp1' : (-4.0, 3.5),
        'AFp1' : (-1.5, 3.5),
        'AFp2' : (1.5, 3.5),
        'Fp2' : (4.0, 3.5),
        'AF7' : (-4.0, 3.0),
        'AF5' : (-3.0, 3.0),
        'AF3' : (-2.0, 3.0),
        'AFz' : (0.0, 3.0),
        'AF4' : (2.0, 3.0),
        'AF6' : (3.0, 3.0),
        'AF8' : (4.0, 3.0),
        'FAF5' : (-2.5, 2.5),
        'FAF1' : (-0.65, 2.5),
        'FAF2' : (0.65, 2.5),
        'FAF6' : (2.5, 2.5),
        'F9' : (-5.0, 2.0),
        'F7' : (-4.0, 2.0),
        'F5' : (-3.0, 2.0),
        'F3' : (-2.0, 2.0),
        'F1' : (-1.0, 2.0),
        'Fz' : (0.0, 2.0),
        'F2' : (1.0, 2.0),
        'F4' : (2.0, 2.0),
        'F6' : (3.0, 2.0),
        'F8' : (4.0, 2.0),
        'F10' : (5.0, 2.0),
        'FFC9' : (-4.5, 1.5),
        'FFC7' : (-3.5, 1.5),
        'FFC5' : (-2.5, 1.5),
        'FFC3' : (-1.5, 1.5),
        'FFC1' : (-0.5, 1.5),
        'FFC2' : (0.5, 1.5),
        'FFC4' : (1.5, 1.5),
        'FFC6' : (2.5, 1.5),
        'FFC8' : (3.5, 1.5),
        'FFC10' : (4.5, 1.5),
        'FT9' : (-5.0, 1.0),
        'FT7' : (-4.0, 1.0),
        'FC5' : (-3.0, 1.0),
        'FC3' : (-2.0, 1.0),
        'FC1' : (-1.0, 1.0),
        'FCz' : (0.0, 1.0),
        'FC2' : (1.0, 1.0),
        'FC4' : (2.0, 1.0),
        'FC6' : (3.0, 1.0),
        'FT8' : (4.0, 1.0),
        'FT10' : (5.0, 1.0),
        'CFC9' : (-4.5, 0.5),
        'CFC7' : (-3.5, 0.5),
        'CFC5' : (-2.5, 0.5),
        'CFC3' : (-1.5, 0.5),
        'CFC1' : (-0.5, 0.5),
        'CFC2' : (0.5, 0.5),
        'CFC4' : (1.5, 0.5),
        'CFC6' : (2.5, 0.5),
        'CFC8' : (3.5, 0.5),
        'CFC10' : (4.5, 0.5),
        'T9' : (-5.0, 0.0),
        'T7' : (-4.0, 0.0),
        'C5' : (-3.0, 0.0),
        'C3' : (-2.0, 0.0),
        'C1' : (-1.0, 0.0),
        'Cz' : (0.0, 0.0),
        'C2' : (1.0, 0.0),
        'C4' : (2.0, 0.0),
        'C6' : (3.0, 0.0),
        'T8' : (4.0, 0.0),
        'T10' : (5.0, 0.0),
        'A1' : (-5.0, -0.5),
        'CCP7' : (-3.5, -0.5),
        'CCP5' : (-2.5, -0.5),
        'CCP3' : (-1.5, -0.5),
        'CCP1' : (-0.5, -0.5),
        'CCP2' : (0.5, -0.5),
        'CCP4' : (1.5, -0.5),
        'CCP6' : (2.5, -0.5),
        'CCP8' : (3.5, -0.5),
        'A2' : (5.0, -0.5),
        'TP9' : (-5.0, -1.0),
        'TP7' : (-4.0, -1.0),
        'CP5' : (-3.0, -1.0),
        'CP3' : (-2.0, -1.0),
        'CP1' : (-1.0, -1.0),
        'CPz' : (0.0, -1.0),
        'CP2' : (1.0, -1.0),
        'CP4' : (2.0, -1.0),
        'CP6' : (3.0, -1.0),
        'TP8' : (4.0, -1.0),
        'TP10' : (5.0, -1.0),
        'PCP9' : (-4.5, -1.5),
        'PCP7' : (-3.5, -1.5),
        'PCP5' : (-2.5, -1.5),
        'PCP3' : (-1.5, -1.5),
        'PCP1' : (-0.5, -1.5),
        'PCP2' : (0.5, -1.5),
        'PCP4' : (1.5, -1.5),
        'PCP6' : (2.5, -1.5),
        'PCP8' : (3.5, -1.5),
        'PCP10' : (4.5, -1.5),
        'P9' : (-5.0, -2.0),
        'P7' : (-4.0, -2.0),
        'P5' : (-3.0, -2.0),
        'P3' : (-2.0, -2.0),
        'P1' : (-1.0, -2.0),
        'Pz' : (0.0, -2.0),
        'P2' : (1.0, -2.0),
        'P4' : (2.0, -2.0),
        'P6' : (3.0, -2.0),
        'P8' : (4.0, -2.0),
        'P10' : (5.0, -2.0),
        'PPO7' : (-4.5, -2.5),
        'PPO5' : (-3.0, -2.5),
        'PPO3' : (-2.0, -2.5),
        'PPO1' : (-0.65, -2.5),
        'PPO2' : (0.65, -2.5),
        'PPO4' : (2.0, -2.5),
        'PPO6' : (3.0, -2.5),
        'PPO8' : (4.5, -2.5),
        'PO9' : (-5.5, -2.6),
        'PO7' : (-4.0, -3),
        'PO5' : (-3.0, -3),
        'PO3' : (-2.0, -3),
        'PO1' : (-1.0, -3),
        'POz' : (0.0, -3),
        'PO2' : (1.0, -3),
        'PO4' : (2.0, -3),
        'PO6' : (3.0, -3),
        'PO8' : (4.0, -3),
        'PO10' : (5.5, -2.6),
        'OPO1' : (-1.5, -3.5),
        'OPO2' : (1.5, -3.5),
        'O9' : (-6.5, -3.5),
        'O1' : (-4.0, -3.5),
        'O2' : (4.0, -3.5),
        'O10' : (6.5, -3.5),
        'Oz' : (0.0, -4.0),
        'OI1' : (1.5, -4.5),
        'OI2' : (-1.5, -4.5),
        'I1' : (1.0, -5),
        'Iz' : (0.0, -5),
        'I2' : (-1, -5)}
    
    # create list with 17 empty lists. one for every potential row of channels.
    channel_lists=[]
    for i in range(18):
        channel_lists.append([])
    
    # distribute the channels to the lists by their y-position
    count = 0
    for c in data.axes[1]:
        if c in tts.channels:
            # entries in channel_lists: (<channel_name>, <x-position>, <position in Data>
            channel_lists[ordering[system[c][1]]].append((c, system[c][0], count))
        count = count + 1
            
    # sort the lists of channels by their x-position
    for l in channel_lists:
        l.sort(key = lambda list: list[1])
    #print(channel_lists)
    
    # calculate the needed dimensions of the grid
    columns = max(map(len, channel_lists))
    rows = 0
    for l in channel_lists:
        if len(l) > 0: rows = rows + 1
    #print("rows: " + str(rows) + ", columns: " + str(columns))
    
    plt.clf()
    gs = gridspec.GridSpec(rows, columns)
    
    row=0
    for l in channel_lists:
        if len(l) > 0:
            for i in range(len(l)):
                col_pos = int(i + ((columns-len(l)) - np.ceil((columns-len(l))/2.)))
                _subplot_timeinterval(data, gs[row, col_pos], epoch=-1, highlights=highlights, legend=True, channel=l[i][2])
                
                # hide the axes
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                
                # at this moment just to show what's what
                #plt.gca().annotate(l[i][0], (20, 20), xycoords='axes pixels')
            row=row+1
    
    # adjust the spacing
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05, hspace=0.2, wspace=0.2)
    
    # saving if specified
    if save:
        if save_path is None:
            plt.savefig(save_name, bbox_inches='tight')
        else:
            plt.savefig(save_path + save_name, bbox_inches='tight')
    
    # showing if specified
    if show: plt.show()
    

# plots the values of a single point of time on a scalp
# data: wyrm.types.Data object containing the data to plot
# time: the point in time to plot. (types.Data.data[time])
# annotate (optional): boolean to switch channell annotations
def plot_scalp(data, time, annotate=True, show=True, save=False, save_name='system_plot', save_path=None):
    plt.clf()
    
    _subplot_scalp(data.data[time], data.axes[1], annotate=annotate)
    
    if show:
        plt.show()
    
    # saving if specified
    if save:
        if save_path is None:
            plt.savefig(save_name, bbox_inches='tight')
        else:
            plt.savefig(save_path + save_name, bbox_inches='tight')
    
    # showing if specified
    if show: plt.show()
    
    
def _subplot_scalp(v, channel, position=None, annotate=True):

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
    
    # add a nose
    plt.plot([-0.1, 0], [0.99, 1.1], 'k-', lw=3)
    plt.plot([0.1, 0], [0.99, 1.1], 'k-', lw=3)
    
    # add ears
    vertsr = [
    (0.99, 0.13),  # P0
    (1.10, 0.3), # P1
    (1.10, -0.3), # P2
    (0.99, -0.13), # P3
    ]
    
    vertsl = [
    (-0.99, 0.13),  # P0
    (-1.10, 0.3), # P1
    (-1.10, -0.3), # P2
    (-0.99, -0.13), # P3
    ]
    
    codes = [Path.MOVETO,
         Path.CURVE4,
         Path.CURVE4,
         Path.CURVE4,
         ]

    pathr = Path(vertsr, codes)
    pathl = Path(vertsl, codes)
    patchr = patches.PathPatch(pathr, facecolor='none', lw=2)
    patchl = patches.PathPatch(pathl, facecolor='none', lw=2)
    plt.gca().add_patch(patchr)
    plt.gca().add_patch(patchl)
    
    # add markers at channel positions
    plt.plot(x, y, 'k+', ms=15, mew=1.5)
    
    # set the axes limits, so the figure is centered on the scalp
    plt.gca().set_ylim([-1.3, 1.3])
    plt.gca().set_xlim([-1.4, 1.4])
    
    if annotate:
        for i in zip(channel, zip(x,y)):
            plt.annotate(" " + i[0], i[1])

    
# adds a timeinterval subplot to the current figure at the specified position.
# data: wyrm.types.Data
# position: position of the subplot
# epoch: specifies the epoch to plot
# highlights (optional): a wyrm.plot.Highlight object to create highlights
# legend (optional): boolean to switch the legend on or off 
# channel (optional): used for plotting only one specific channel
def _subplot_timeinterval(data, position, epoch, highlights=None, legend=True, channel=None):
    
    # plotting of the data
    plt.subplot(position)
    
    # epoch is -1 when there are no epochs
    if(epoch == -1):
        if channel is None:
            plt.plot(data.axes[0], data.data)
        else:
            plt.plot(data.axes[0], data.data[:, channel])
    else:
        if channel is None:
            plt.plot(data.axes[len(data.axes) - 2], data.data[epoch])
        else:
            plt.plot(data.axes[len(data.axes) - 2], data.data[epoch, channel])
    
    # plotting of highlights
    set_highlights(highlights, axes=[plt.gca()])
    
    # labeling of channels
    if legend:
        if channel is None:
            plt.legend(data.axes[len(data.axes) - 1])
        else:
            plt.legend([data.axes[len(data.axes) - 1][channel]])
    
    plt.grid(True)
    

# Adds highlights to the specified axes.
# obj_highlight: an instance of the Highlight class
# axes (optional): a list of axes
def set_highlights(obj_highlight, axes=None):
    
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
# draw (optional): boolean to switch immediate drawing  
def set_labels(xLabel, yLabel, axes=None, draw=True):
    
    if axes is None:
        axes = plt.gcf().axes
        
    # labeling of axes
    for ax in axes:
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel, rotation = 0)
        
    if draw: plt.draw()
        

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
