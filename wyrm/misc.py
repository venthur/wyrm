#!/usr/bin/env python


from __future__ import division

from os import path
import logging
import re

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import signal

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


"""
Three Kinds of EEG Data
-----------------------

1. Raw: A numpy array (time x channel)

2. Continous Data: An object holding raw data together with meta information,
like the sampling frequency, channel and the marker

3. Epoched Data: An object holding a list of Continuous Data
"""

class Cnt(object):
    """Continuous Data Object.

    This object represents a stream of continuous EEG data. It is
    defined by the raw EEG data (samples x channels), the sampling
    frequency, the channel names and the marker.

    Parameters
    ----------
    data : ndarray (time, channel)
        The raw EEG data in a 2 dimensional nd array (sample, channel)
    fs : float
        The sampling frequency
    channel : array of strings
        The channel names in the same order as they appear in `data`
    marker : array of ???

    Attributes
    ----------
    data : numpy array (samples x channels)
        Defines the raw EEG data.

    fs : float
        The sampling frequency of the given data set

    channel : numpy array of strings
        The names of the channels in the same order as they appear in data

    marker : ???

    """
    def __init__(self, data, fs, channel, marker):
        self.data = data
        self.fs = fs
        self.channel = np.array(channel)
        self.marker = marker
        # TODO: should we make some sanity checks here?

    # Should this class only be a wrapper for the raw data + meta info?
    # Should it provide its own methods for removing channels, resampling, etc
    # or should this be done by mushu library methods?


class Epo(object):

    def __init__(self, data):
        self.data = data


def select_channels(cnt, regexp_list, invert=False):
    """Select channels from data.

    The matching is case-insensitive and locale-aware (as in re.IGNORECASE and
    re.LOCALE). The regular expression always has to match the whole channel
    name string

    Parameters
    ----------
    cnt : Continuous data

    regexp_list : Array of regular expressions
        The regular expressions provided, are used directly by Python's
        :mod:`re` module, so all regular expressions which are understood
        by this module are allowed.

        Internally the :func:`re.match` method is used, additionally to
        check for a match (which also matches substrings), it is also
        checked if the whole string matched the pattern.

    invert : Boolean (default=False)
        If True the selection is inverted. Instead of selecting specific
        channels, you are removing the channels.

    Returns
    -------
    cnt
        A copy of the continuous data with the channels, matched by the list of
        regular expressions.

    Examples
    --------
    Select all channels Matching 'af.*' or 'fc.*'

    >>> cnt_new = select_channels(cnt, ['af.*', 'fc.*'])

    Remove all channels Matching 'emg.*' or 'eog.*'

    >>> cnt_new = select_channels(cnt, ['emg.*', 'eog.*'], invert=True)

    Even if you only provide one Regular expression, it has to be in an
    array:

    >>> cnt_new = select_channels(cnt, ['af.*'])

    See Also
    --------
    re : Python's Regular Expression module for more information about regular
        expressions.

    """
    # TODO: make it work with epos
    chan_mask = np.array([False for i in range(len(cnt.channel))])
    for c_idx, c in enumerate(cnt.channel):
        for regexp in regexp_list:
            m = re.match(regexp, c, re.IGNORECASE | re.LOCALE)
            if m and m.group() == c:
                chan_mask[c_idx] = True
                # no need to look any further for matches for this channel
                break
    if invert:
        chan_mask = ~chan_mask
    data = cnt.data[:,chan_mask]
    channel = cnt.channel[chan_mask]
    return Cnt(data, cnt.fs, channel, cnt.marker)


def load_brain_vision_data(vhdr):
    """Load Brain Vision data from a file.

    This methods loads the continuous EEG data, the channel names, sampling
    frequency and the marker.

    Parameters
    ----------
    vhdr : str
        Path to a VHDR file

    Returns
    -------
    cnt
        Continuous Data.

    """
    logger.debug('Loading Brain Vision Data Exchange Header File')
    with open(vhdr) as fh:
        fdata = map(str.strip, fh.readlines())
    fdata = filter(lambda x: not x.startswith(';'), fdata)
    fdata = filter(lambda x: len(x) > 0, fdata)
    # check for the correct file version:
    assert fdata[0].endswith('1.0')
    # read all data into a dict where the key is the stanza of the file
    file_dict = dict()
    for line in fdata[1:]:
        if line.startswith('[') and line.endswith(']'):
            current_stanza = line[1:-1]
            file_dict[current_stanza] = []
        else:
            file_dict[current_stanza].append(line)
    # translate known stanzas from simple list of strings to a dict
    for stanza in 'Common Infos', 'Binary Infos', 'Channel Infos':
        logger.debug(stanza)
        file_dict[stanza] = {line.split('=', 1)[0]: line.split('=', 1)[1] for line in file_dict[stanza]}
    # now file_dict contains the parsed data from the vhdr file
    # load the rest
    data_f = file_dict['Common Infos']['DataFile']
    marker_f = file_dict['Common Infos']['MarkerFile']
    data_f = path.sep.join([path.dirname(vhdr), data_f])
    marker_f = path.sep.join([path.dirname(vhdr), marker_f])
    n_channels = int(file_dict['Common Infos']['NumberOfChannels'])
    sampling_interval_microseconds = float(file_dict['Common Infos']['SamplingInterval'])
    fs = 1 / (sampling_interval_microseconds / 10**6)
    channels = [file_dict['Channel Infos']['Ch%i' % (i + 1)] for i in range(n_channels)]
    channels = map(lambda x: x.split(',')[0], channels)
    # some assumptions about the data...
    assert file_dict['Common Infos']['DataFormat'] == 'BINARY'
    assert file_dict['Common Infos']['DataOrientation'] == 'MULTIPLEXED'
    assert file_dict['Binary Infos']['BinaryFormat'] == 'INT_16'
    print fs, n_channels
    print marker_f
    print data_f
    print channels
    # load EEG data
    logger.debug('Loading EEG Data.')
    data = np.fromfile(data_f, np.int16)
    data = data.reshape(-1, n_channels)
    # load marker
    logger.debug('Loading Marker.')
    regexp = r'^Mk(?P<mrk_nr>[0-9]*)=.*,(?P<mrk_descr>.*),(?P<mrk_pos>[0-9]*),[0-9]*,[0-9]*$'
    mrk = []
    with open(marker_f) as fh:
        for line in fh:
            line = line.strip()
            match = re.match(regexp, line)
            if match is None:
                continue
            mrk_pos = match.group('mrk_pos')
            mrk_descr = match.group('mrk_descr')
            if len(mrk_descr) > 1:
                mrk.append([mrk_pos, mrk_descr])
    return data, mrk, channels, fs


def plot_channels(data, n_channels):
    ax = []
    for i in range(n_channels):
        if i == 0:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1)
        else:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1, sharex=ax[0], sharey=ax[0])
        ax.append(a)
        a.plot(data[:, i])
        a.set_title(channels[i])


def segmentation(data, mrk, start, end):
    data2 = []
    for i in mrk:
        i_start, i_end = i+start, i+end
        chunk = data[i_start:i_end]
        data2.append(chunk)
    return np.array(data2)


def filter_bp(data, fs, low, high):
    # band pass filter the data
    fs_n = fs * 0.5
    #logger.debug('Calculating butter order...')
    #butter_ord, f_butter = signal.buttord(ws=[(low - .1) / fs_n, (high + .1) / fs_n],
    #                                      wp=[low / fs_n, high / fs_n],
    #                                      gpass=0.1,
    #                                      gstop=3.0
    #                                      )

    #logger.debug("{ord} {fbutter} {low} {high}".format(**{'ord': butter_ord,
    #                                                      'fbutter': f_butter,
    #                                                      'low': low / fs_n,
    #                                                      'high': high / fs_n}))
    butter_ord = 4
    b, a = signal.butter(butter_ord, [low / fs_n, high / fs_n], btype='band')
    return signal.lfilter(b, a, data, axis=0)


def calculate_csp(class1, class2):
    """Calculate the Common Spatial Pattern (CSP) for two classes.

    Examples
    --------
    Calculate the CSP for two classes::

    >>> w, a, d = calculate_csp(c1, c2)

    Take the first two and the last two columns of the sorted filter::

    >>> w = w[:, (0, 1, -2, -1)]

    Apply the new filter to your data d of the form (time, channels)::

    >>> filtered = np.dot(d, w)

    You'll probably want to get the log-variance along the time axis::

    >>> filtered = np.log(np.var(filtered, 0))

    This should result in four numbers (one for each channel).

    Parameters
    ----------
    class1 :
        A matrix of the form (trials, time, channels) representing class 1.
    class2 :
        A matrix of the form (trials, time, channels) representing the second
        class.

    Returns
    -------
    A tuple (v, a, d). You should use the columns of the matrices, where

        v:
            The sorted spacial filters.
        a:
            The sorted spacial patterns (i.e. column x of a represents
            the pattern of column x of v.
        d:
            The variances of the components.

    References
    ----------
    http://en.wikipedia.org/wiki/Common_spatial_pattern

    """
    # sven's super simple matlab code
    # function [W, A, lambda] = my_csp(X1, X2)
    #     % compute covariance matrices of the two classes
    #     C1 = compute_Covariance_Matrix(X1);
    #     C2 = compute_Covariance_Matrix(X2);
    #     % solution of CSP objective via generalized eigenvalue problem
    #     [W, D] = eig(C1-C2, C1+C2);
    #     % make sure the eigenvalues and eigenvectors are sorted correctly
    #     [lambda, sort_idx] = sort(diag(D), 'descend');
    #     W = W(:,sort_idx);
    #     A = inv(W)';

    n_channels = class1.shape[2]
    # we need a matrix of the form (observations, channels) so we stack trials
    # and time per channel together
    x1 = class1.reshape(-1, n_channels)
    x2 = class2.reshape(-1, n_channels)
    # compute covariance matrices of the two classes
    c1 = np.cov(x1.transpose())
    c2 = np.cov(x2.transpose())
    # solution of csp objective via generalized eigenvalue problem
    # in matlab the signature is v, d = eig(a, b)
    d, v = sp.linalg.eig(c1-c2, c1+c2)
    d = d.real
    # make sure the eigenvalues and -vectors are correctly sorted
    indx = np.argsort(d)
    # reverse
    indx = indx[::-1]
    d = d.take(indx)
    v = v.take(indx, axis=1)
    a = sp.linalg.inv(v).transpose()
    return v, a, d


# TODO: use that method
def moving_average(data, ws):
    window = numpy.ones(ws) / float(ws)
    return np.convolve(data, window, 'same')

