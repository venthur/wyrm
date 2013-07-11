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



#Three Kinds of EEG Data
#-----------------------
#
#1. Raw: A numpy array (time x channel)
#
#2. Continous Data: An object holding raw data together with meta information,
#like the sampling frequency, channel and the marker
#
#3. Epoched Data: An object holding a list of Continuous Data


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
    channels : array of strings
        The channel names in the same order as they appear in `data`
    markers : array of ???

    Attributes
    ----------
    data : numpy array (samples x channels)
        Defines the raw EEG data.
    fs : float
        The sampling frequency of the given data set
    channels : numpy array of strings
        The names of the channels in the same order as they appear in data
    markers: ???

    """
    def __init__(self, data, fs, channels, markers):
        self.data = data
        self.fs = fs
        self.channels= np.array(channels)
        self.markers= markers
        # TODO: should we make some sanity checks here?
    # Should this class only be a wrapper for the raw data + meta info?
    # Should it provide its own methods for removing channels, resampling, etc
    # or should this be done by mushu library methods?

class Epo(object):
    """Epoched data object.

    An Epoch represents a list of Continuous data. Each element `i` of
    an Epoch is assigned to a class `c = classes[i]` of the name
    `classname[c]`.

    Parameters
    ----------
    data : ndarray (epoch, sample, channel)
        The raw, epoched EEG data in a 3 dimensional ndarray (epoch,
        sample, channel)
    fs : float
        The sampling frequency
    channels : array of strings
        The channel names in the same order as they appear in `data`
    markers : ???
        Not defined yet!
    classes : array
        A 1 dimensional array, each entry represents the class for the
        respective epoch in `data`. The value is also the index of
        `class_names` for a human readable description of the class.
    class_names : array of strings
        The human readable class names. The indices of the classes in
        `class_names` match the values in `classes`.


    Attributes
    ----------
    data : ndarray (epoch, sample, channel)
        The raw and epoched EEG data: (epochs, samples, channels).
    fs : float
        The sampling frequency
    channels : array of strings
        The channel names in the same order as they appear in `data`
    markers : NOT DEFINED YET
        Not defined yet!
    classes : list
        A 1 dimensional array, each entry represents the class for the
        respective epoch in `data`. The value is also the index of
        `class_names` for a human readable description of the class.
    class_names : array of strings
        The human readable class names. The indices of the classes in
        `class_names` match the values in `classes`.

    """
    def __init__(self, data, fs, channels, markers, classes, class_names):
        self.data = data
        self.fs = fs
        self.channels = np.array(channels)
        self.markers = markers
        self.classes = np.array(classes)
        self.class_names = class_names


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
    remove_channels : Remove Channels
    re : Python's Regular Expression module for more information about regular
        expressions.

    """
    # TODO: make it work with epos
    chan_mask = np.array([False for i in range(len(cnt.channels))])
    for c_idx, c in enumerate(cnt.channels):
        for regexp in regexp_list:
            m = re.match(regexp, c, re.IGNORECASE | re.LOCALE)
            if m and m.group() == c:
                chan_mask[c_idx] = True
                # no need to look any further for matches for this channel
                break
    if invert:
        chan_mask = ~chan_mask
    data = cnt.data[:,chan_mask]
    channels = cnt.channels[chan_mask]
    return Cnt(data, cnt.fs, channels, cnt.markers)


def remove_channels(cnt, regexp_list):
    """Remove channels from data.

    This method just calls :func:`select_channels` with the `invert`
    parameter set to `True`.

    Returns
    -------
    Cnt
        A copy of the cnt with the channels removed.

    See Also
    --------

    select_channels: Select Channels

    """
    return select_channels(cnt, regexp_list, invert=True)


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
                mrk.append([int(mrk_pos), mrk_descr])
    return Cnt(data, fs, channels, mrk)


def plot_channels(cnt):
    ax = []
    n_channels = len(cnt.channels)
    for i, chan in enumerate(cnt.channels):
        if i == 0:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1)
        else:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1, sharex=ax[0], sharey=ax[0])
        ax.append(a)
        a.plot(cnt.data[:, i])
        a.set_title(chan)


def cnt_to_epo(cnt, marker_def, ival):
    """

    Parameters
    ----------
    cnt : Continuous data object
    marker_def : dict
        The keys are class names, the values are lists of markers
    ival : [int, int]
        The interval in milliseconds to cut around the markers. I.e. to
        get the interval starting with the marker plus the remaining
        100ms define the interval like [0, 100].

        To get 200ms before the marker until 100ms after the marker do:
        [-200, 100]

        Only negative or positive values are possible (i.e. [-500, -100])

    Returns
    -------
    epo
        The resulting epoched data.


    Examples
    --------

    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # Epoch the data -500ms and +700ms around the markers defined in
    >>> # md
    >>> epo = cnt_to_epo(cnt, md, [-500, 700])

    """
    assert ival[0] <= ival[1]
    # calculate the number of samples, given the fs of the cnt
    factor = cnt.fs / 1000
    start, stop = [i * factor for i in ival]
    data = []
    classes = []
    class_names = sorted(marker_def.keys())
    for pos, m in cnt.markers:
        pos = int(pos)
        for class_idx, classname in enumerate(class_names):
            if m in marker_def[classname]:
                chunk = cnt.data[pos+start: pos+stop]
                data.append(chunk)
                classes.append(class_idx)
    # convert the array of cnts into an (epo, time, channel) array
    data = np.array(data)
    epo = Epo(data, cnt.fs, cnt.channels, cnt.markers, classes, class_names)
    return epo


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


def subsample(cnt, factor):
    """Subsample the data by factor `factor`.

    This method subsamples by taking every `factor` th element starting
    with the first one.

    Note that this method does not low-pass filter the data before
    sub-sampling.

    Parameters
    ----------
    cnt : Cnt
    factor : int

    Returns
    -------
    Cnt

    See Also
    --------
    band_pass

    Examples
    --------

    Load some EEG data with 1kHz, bandpass filter it and downsample it
    by 10 so the resulting sampling frequency is 100Hz.

    >>> cnt = load_brain_vision_data('some/path')
    >>> cnt.fs
    1000.0
    >>> cnt = band_pass(cnt, 8, 40)
    >>> cnt = subsample(cnt, 10)
    >>> cnt.fs
    100.0

    """
    data = cnt.data[..., ::factor, :]
    fs = cnt.fs / factor
    markers = map(lambda x: [int(x[0] / factor), x[1]], cnt.markers)
    return Cnt(data, fs, cnt.channels, markers)

    
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
    class1
        A matrix of the form (trials, time, channels) representing class 1.
    class2
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


def calculate_classwise_average(epo):
    """Calculate the classwise average.

    This method calculates the average continuous per class for all
    classes defined in the `epo`. In other words, if you have two
    different classes, with many continuous data per class, this method
    will calculate the average time course for each class and channel.

    Parameters
    ----------
    epo : Epo

    Returns
    -------
    Epo
        An Epo object holding a continuous per class.

    Examples
    --------

    Split existing continuous data into two classes and calculate the
    average continuous for each class.

    >>> mrk_def = {'std': ['S %2i' % i for i in range(2, 7)],
    ...            'dev': ['S %2i' % i for i in range(12, 17)]
    ...           }
    >>> epo = misc.cnt_to_epo(cnt, mrk_def, [0, 660])
    >>> avg_epo = calculate_classwise_average(epo)
    >>> plot(avg_epo.data[0])
    >>> plot(avg_epo.data[1])

    """
    data = []
    classes = []
    classnames = []
    for i, classname in enumerate(epo.class_names):
        avg = np.average(epo.data[epo.classes == i], axis=0)
        classes.append(i)
        classnames.append(classname)
        data.append(avg)
    classes = np.array(classes)
    data = np.array(data)
    return Epo(data, epo.fs, epo.channels, epo.markers, classes, classnames)


