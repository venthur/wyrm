"""Methods for creation on testdata.

This module contains various methods for creating random test data to try out the plotting features.

"""

import numpy as np
import random as rnd
import tentensystem as tts
from wyrm.types import Data
import wyrm.plot as p


# automated creation of test data
def _create_channel(steps=100):
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


# test function: (x^3 * cos(x)) / 20, x in [-5, -2] (with variations)
def create_data_ti(channel_count=2, steps=100):
    
    data = np.zeros([steps, channel_count])
    channels = []
    for i in range(channel_count):
        data[:, i] = _create_channel(steps)
        channels.append('ch' + str(i))
        
    axes = [np.arange(0, steps*10, 10), channels]
    names = ["time", "channel"]
    units = ["ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return dat


def create_data_all():
    d = create_data_ti(channel_count=142)
    d.axes[1] = np.array(tts.channels.keys())
    return d


def create_data_scalp(t=0):
    d = create_data_ti(channel_count=142)
    chans = np.array(tts.channels.keys())
    return d.data[t], chans


def create_data_some():
    d = create_data_ti(channel_count=25)
    d.axes[1] = np.array(['Fpz', 'AF7', 'AF8', 'FFC9', 'FFC10', 'AF5', 'AF6', 'FFC5', 'FFC6', 'T7', 'T8', 'C3', 'C4',
                          'Cz', 'TP7', 'TP8', 'P1', 'P2', 'P7', 'P8', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz'])
    return d


def create_epoched_data_ti(epoch_count=4, channel_count=2, steps=100):
    data = np.zeros([epoch_count, steps, channel_count])
    for i in range(epoch_count):
        for o in range(channel_count):
            data[i, :, o] = _create_channel(steps)
            
    # create the channel labels
    channels = []
    for i in range(channel_count):
        channels.append('ch' + str(i))
        
    # create the class labels
    classes = []
    for i in range(epoch_count):
        classes.append('class' + str(i % 2))
        
    axes = [classes, np.arange(0, steps*10, 10), channels]
    names = ["class", "time", "channel"]
    units = ["class_stuff", "ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return dat


def create_tenten_data(channel_count=21, steps=100):
    data = np.zeros([steps, channel_count])
    #channels = []
    chan_names = ['A1', 'A2', 'C3', 'C4', 'Cz', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'O1', 'O2', 'P3', 'P4',
                  'Pz', 'T3', 'T4', 'T5', 'T6']
         
    for i in range(channel_count):
        data[:, i] = _create_channel(steps)
        
    if channel_count <= len(chan_names):
        channels = chan_names[:channel_count]
    else:
        channels = chan_names
        i = len(chan_names)
        while i < channel_count:
            channels.append('ch' + str(i))
            i += 1

    axes = [np.arange(0, steps*10, 10), channels]
    names = ["time", "channel"]
    units = ["ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return dat


def create_tenten_data_rnd(channel_count=20, steps=100):
    sys = p._get_system().keys()

    if channel_count > len(sys):
        channel_count = len(sys)

    channels = []
    while len(channels) < channel_count:
        c = sys[rnd.randrange(0, len(sys))]
        if c not in channels:
            channels.append(c)

    data = np.zeros([steps, channel_count])
    for i in range(channel_count):
        data[:, i] = _create_channel(steps)

    axes = [np.arange(0, steps*10, 10), channels]
    names = ["time", "channel"]
    units = ["ms", "channel_stuff"]
    dat = Data(data, axes, names, units)
    return dat


def grid_test(cols=4, rows=3, hpad=.05, vpad=.05):
    g = p.calc_grid(cols, rows, hpad, vpad)
    p.plt.figure()
    for r in g:
        p._subplot_timeinterval(create_data_ti(), r, -1)
    p.plt.show()