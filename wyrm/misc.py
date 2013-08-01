#!/usr/bin/env python

"""Miscellaneous toolbox methods.

This module currently contains all the processing methods.

The contents of this module should move into various other modules once
it is ready for prime.

"""



from __future__ import division

from os import path
import logging
import re

import numpy as np
import scipy as sp
from scipy import signal

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


# Three Kinds of EEG Data
# -----------------------
#
# 1. Raw: A numpy array (time x channel)
#
# 2. Continous Data: An object holding raw data together with meta information,
# like the sampling frequency, channel and the marker
#
# 3. Epoched Data: An object holding a list of Continuous Data

class Cnt(object):
    """Continuous Data Object.

    This object represents a stream of continuous EEG data. It is
    defined by the raw EEG data (samples x channels), the sampling
    frequency, the channel names and the marker.

    Parameters
    ----------
    data : 2darray
        The raw EEG data in a 2 dimensional nd array (sample, channel)
    fs : float
        The sampling frequency
    channels : array of strings
        The channel names in the same order as they appear in ``data``
    markers : array of (int, str)
        the int represents the position in data this marker belongs to,
        and str is the actual marker

    Attributes
    ----------
    data : 2darray
        Defines the raw EEG data (sample, channel)
    fs : float
        The sampling frequency of the given data set
    channels : numpy array of strings
        The names of the channels in the same order as they appear in data
    markers : array of (int, str)
        the int represents the position in data this marker belongs to,
        and str is the actual marker
    t : array of float
        the time in ms for each element in data.

    """
    def __init__(self, data, fs, channels, markers):
        self.data = data
        self.fs = fs
        self.channels = np.array(channels)
        self.markers = markers
        duration = 1000 * data.shape[0] / fs
        self.t = np.linspace(0, duration, data.shape[0], endpoint=False)
        # TODO: should we make some sanity checks here?


class Epo(object):
    """Epoched data object.

    An Epoch represents a list of Continuous data. Each element ``i`` of
    an Epoch is assigned to a class ``c = classes[i]`` of the name
    ``classname[c]``.

    Each cnt of this epo has the same length (number of samples), number
    of channels and time interval.

    Parameters
    ----------
    data : ndarray (epoch, sample, channel)
        The raw, epoched EEG data in a 3 dimensional ndarray (epoch,
        sample, channel)
    fs : float
        The sampling frequency
    channels : array of strings
        The channel names in the same order as they appear in ``data``
    markers : array of arrays of (int, str)
        for each epoch there is an array of mackers (int, str), where
        the int indicates the position of the marker relative to data
        and str is the actual marker.
    classes : array
        A 1 dimensional array, each entry represents the class for the
        respective epoch in ``data``. The value is also the index of
        ``class_names`` for a human readable description of the class.
    class_names : array of strings
        The human readable class names. The indices of the classes in
        ``class_names`` match the values in ``classes``.
    t_start : float
        (start) time in ms of the interval in relation to the event of
        the epoch (e.g. -100, 0, or 200)


    Attributes
    ----------
    data : (N, N, N) ndarray
        The raw and epoched EEG data: (epochs, samples, channels).
    fs : float
        The sampling frequency
    channels : array of strings
        The channel names in the same order as they appear in ``data``
    markers : array of arrays of (int, str)
        for each epoch there is an array of mackers (int, str), where
        the int indicates the position of the marker relative to data
        and str is the actual marker.
    classes : list
        A 1 dimensional array, each entry represents the class for the
        respective epoch in ``data``. The value is also the index of
        ``class_names`` for a human readable description of the class.
    class_names : array of strings
        The human readable class names. The indices of the classes in
        ``class_names`` match the values in ``classes``.
    t : (N,) nd array
        the time in ms for each element in data

    """
    def __init__(self, data, fs, channels, markers, classes, class_names, t_start):
        self.data = data
        self.fs = fs
        self.channels = np.array(channels)
        self.markers = markers
        self.classes = np.array(classes)
        self.class_names = np.array(class_names)
        duration = 1000 * data.shape[-2] / fs
        self.t = np.linspace(t_start, t_start + duration, data.shape[-2], endpoint=False)

    def __getitem__(self, key):
        data = self.data[key]
        cnt = Cnt(data, self.fs, self.channels, self.markers[key])
        cnt.t = self.t
        return cnt

def select_channels(cnt, regexp_list, invert=False):
    """Select channels from data.

    The matching is case-insensitive and locale-aware (as in
    ``re.IGNORECASE`` and ``re.LOCALE``). The regular expression always
    has to match the whole channel name string

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
    cnt : Cnt
        A copy of the continuous data with the channels, matched by the
        list of regular expressions.

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
    re : Python's Regular Expression module for more information about
        regular expressions.

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

    This method just calls :func:`select_channels` with the ``invert``
    parameter set to ``True``.

    Returns
    -------
    cnt : Cnt
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
    cnt : Cnt
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


def segment_cnt(cnt, marker_def, ival):
    """Convert a continuous data object to an peoched one.

    Given a continuous data object, a definition of classes, and an
    interval, this method looks for markers as defined in ``marker_def``
    and slices the cnt according to the time interval given with
    ``ival``.  The returned ``Epo`` object stores those slices and the
    class each slice belongs to.


    Parameters
    ----------
    cnt : Cnt
    marker_def : dict
        The keys are class names, the values are lists of markers
    ival : [int, int]
        The interval in milliseconds to cut around the markers. I.e. to
        get the interval starting with the marker plus the remaining
        100ms define the interval like [0, 100].

        To get 200ms before the marker until 100ms after the marker do:
        ``[-200, 100]``

        Only negative or positive values are possible (i.e. ``[-500,
        -100]``)

    Returns
    -------
    epo : Epo
        The resulting epoched data.


    Examples
    --------
    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # Epoch the data -500ms and +700ms around the markers defined in
    >>> # md
    >>> epo = segment_cnt(cnt, md, [-500, 700])

    See Also
    --------
    Epo

    """
    assert ival[0] <= ival[1]
    data = []
    classes = []
    class_names = sorted(marker_def.keys())
    # create an marker array similar to .data
    marker = ['' for i in range(cnt.t.shape[0])]
    for pos, txt in cnt.markers:
        marker[pos] = txt
    marker = np.array(marker)
    markers = []
    for pos, m in cnt.markers:
        t = cnt.t[int(pos)]
        for class_idx, classname in enumerate(class_names):
            if m in marker_def[classname]:
                mask = np.logical_and(t+ival[0] <= cnt.t, cnt.t <= t+ival[1])
                data.append(cnt.data[mask])
                classes.append(class_idx)
                mrk = []
                for idx, val in enumerate(marker[mask]):
                    if val != '':
                        mrk.append([idx, val])
                markers.append(mrk)
    # convert the array of cnts into an (epo, time, channel) array
    data = np.array(data)
    epo = Epo(data, cnt.fs, cnt.channels, markers, classes, class_names, ival[0])
    return epo


def band_pass(cnt, low, high):
    """Band pass filter the data.

    Parameters
    ----------
    cnt : Cnt
    low, high : int
        the low, and high borders of the desired frequency band

    Returns
    -------
    cnt : Cnt
        the band pass filtered data

    """
    # band pass filter the data
    fs_n = cnt.fs * 0.5
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
    data = signal.lfilter(b, a, cnt.data, axis=-2)
    return Cnt(data, cnt.fs, cnt.channels, cnt.markers)


def select_ival(epo, ival):
    """Select interval from epoched data.

    This method selects the time segment(s) defined by ival in a new Epo
    instance.

    Parameters
    ----------
    epo : Epo
    ival : (float, float)
        Start and end in milliseconds. Start and End are included.

    Returns
    -------
    epo : Epo

    Raises
    ------
    AssertionError
        if the given interval does not fit into ``epo.ival`` or
        ``ival[0] > ival[1]``.

    Examples
    --------

    Select the first 200ms of the epoched data:

    >>> epo2 = select_ival(epo, [0, 200])
    >>> print epo2.t[0], epo2.t[-1]
    0.0 200.0

    """
    assert epo.t[0] <= ival[0] <= epo.t[-1]
    assert epo.t[0] <= ival[1] <= epo.t[-1]
    assert ival[0] <= ival[1]
    mask = np.logical_and(ival[0] <= epo.t, epo.t <= ival[1])
    data = epo.data[..., mask, :]
    return Epo(data, epo.fs, epo.channels, epo.markers, epo.classes, epo.class_names, ival[0])


def select_epochs(epo, indices, invert=False):
    """Select epochs from an Epo object.

    This method selects the epochs with the specified indices.

    Parameters
    ----------
    epo : Epo
    indices : array of ints
        The indices of the elements to select.
    invert : Boolean
        if true keep all elements except the ones defined by ``indices``.

    Returns
    -------
    epo : Epo
        a copy of the epoched data with only the selected epochs included.

    See Also
    --------
    remove_epochs

    Examples
    --------

    Get the first three epochs. Note how the class '2', becomes the only
    class and is thus compressed to '0'.

    >>> epo.classes
    [0, 0, 1, 2, 2]
    >>> epo = select_epochs(epo, [0, 1, 2])
    >>> epo.classes
    [0, 0, 1]

    Remove the fourth epoch

    >>> epo.classes
    [0, 0, 1, 2, 2]
    >>> epo = select_epochs(epo, [3], invert=True)
    >>> epo.classes
    [0, 0, 1, 2]

    """
    mask = np.array([False for i in range(epo.data.shape[0])])
    for i in indices:
        mask[i] = True
    if invert:
        mask = ~mask
    data = epo.data[mask]
    classes = epo.classes[mask]
    return Epo(data, epo.fs, epo.channels, epo.markers, classes, epo.class_names, epo.t[0])


def remove_epochs(epo, indices):
    """Remove epochs from an Epo object.

    This Method just calls :meth:`select_epochs` with the ``inverse``
    paramerter set to ``True``.

    Parameters
    ----------
    epo : Epo
    indices : array of ints
        the indices of the elements to exclude

    Returns
    -------
    epo : Epo

    See Also
    --------
    select_epochs

    """
    return select_epochs(epo, indices, invert=True)


def subsample(cnt, freq):
    """Subsample the data to ``freq`` Hz.

    This method subsamples by taking every ``n`` th element starting
    with the first one and ``n`` being ``cnt.fs / freq``. Please note
    that ``freq`` must be a whole number divisor of ``cnt.fs``.

    Note that this method does not low-pass filter the data before
    sub-sampling.

    Parameters
    ----------
    cnt : Cnt
    freq : float
        the target frequency in Hz

    Returns
    -------
    cnt : Cnt

    See Also
    --------
    band_pass

    Examples
    --------

    Load some EEG data with 1kHz, bandpass filter it and downsample it
    to 100Hz.

    >>> cnt = load_brain_vision_data('some/path')
    >>> cnt.fs
    1000.0
    >>> cnt = band_pass(cnt, 8, 40)
    >>> cnt = subsample(cnt, 100)
    >>> cnt.fs
    100.0

    Raises
    ------
    AssertionError : if ``freq`` is not a whole number divisor of ``cnt.fs``

    """
    assert cnt.fs % freq == 0
    factor = int(cnt.fs / freq)
    data = cnt.data[..., ::factor, :]
    fs = cnt.fs / factor
    markers = map(lambda x: [int(x[0] / factor), x[1]], cnt.markers)
    return Cnt(data, fs, cnt.channels, markers)


def spectrum(cnt):
    """Calculate the normalized spectrum of a continuous data object.


    Returns
    -------
    fourier : ndarray
    freqs : ndarray

    See Also
    --------
    spectrogram, stft

    """
    fourier = np.array([sp.fftpack.rfft(cnt.data[:,i]) for i in range(cnt.data.shape[-1])])
    fourier *= (2 / cnt.data.shape[-2])
    freqs = sp.fftpack.rfftfreq(cnt.data.shape[-2], 1/cnt.fs)
    return fourier, freqs


def spectrogram(cnt):
    """Calculate the spectrogram of a continuous data object.

    See Also
    --------
    spectrum, stft

    """
    framesize = 1000 #ms
    width = int(cnt.fs * (framesize / 1000))
    specgram = np.array([stft(cnt.data[:,i], width) for i in range(cnt.data.shape[-1])])
    freqs = sp.fftpack.rfftfreq(width, 1/cnt.fs)
    return specgram, freqs


def stft(x, width):
    """Short time fourier transform of a real sequence.

    This method performs a discrete short time Fourier transform. It
    uses a sliding window to perform discrete Fourier transforms on the
    data in the Window. The results are returned in an array.

    This method uses a Hanning window on the data in the window before
    calculating the Fourier transform.

    The sliding windows are overlapping by ``width / 2``.

    Parameters
    ----------
    x : ndarray
    width: int
        the width of the sliding window in samples

    Returns
    -------
    fourier : 2d complex array
        the dimensions are time, frequency; the frequencies are evenly
        binned from 0 to f_nyquist

    See Also
    --------
    spectrum, spectrogram, scipy.hanning, scipy.fftpack.rfft

    """
    window = sp.hanning(width)
    fourier = np.array([sp.fftpack.rfft(x[i:i+width] * window) for i in range(0, len(x)-width, width//2)])
    fourier *= (2 / width)
    return fourier


def calculate_csp(class1, class2):
    """Calculate the Common Spatial Pattern (CSP) for two classes.

    You should use the columns of the patterns and filters.

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
    v : 2d array
        the sorted spacial filters
    a : 2d array
        the sorted spacial patterns. Column i of a represents the
        pattern of the filter in column i of v.
    d : 1d array
        the variances of the components


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
    classes defined in the ``epo``. In other words, if you have two
    different classes, with many continuous data per class, this method
    will calculate the average time course for each class and channel.

    Parameters
    ----------
    epo : Epo

    Returns
    -------
    epo : Epo
        An Epo object holding a continuous per class.

    Examples
    --------

    Split existing continuous data into two classes and calculate the
    average continuous for each class.

    >>> mrk_def = {'std': ['S %2i' % i for i in range(2, 7)],
    ...            'dev': ['S %2i' % i for i in range(12, 17)]
    ...           }
    >>> epo = misc.segment_cnt(cnt, mrk_def, [0, 660])
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
    return Epo(data, epo.fs, epo.channels, epo.markers, classes, classnames, epo.t[0])


def correct_for_baseline(epo, ival):
    """Subtract the baseline.

    For each cnt-Element and channel in the given epo, this method
    calculates the average value for the given interval and subtracts
    this value from the channel data.

    Parameters
    ----------
    epo : Epo
    ival : (float, float)
        the start and stop borders in milli seconds. ``ival`` must fit
        into ``epo.ival`` and ``ival[0] <= ival[1]``

    Returns
    -------
    epo : Epo


    Examples
    --------

    Remove the baselines for the interval ``[100, 0]``

    >>> epo = correct_for_baseline(epo, [-100, 0])

    Raises
    ------
    AssertionError
        If the left or right border of ``ival`` is outside of
        ``epo.ival`` or if ``ival`` is malformed.

    """
    # check if ival fits into epo.ival
    assert epo.t[0] <= ival[0] <= epo.t[-1]
    assert epo.t[0] <= ival[1] <= epo.t[-1]
    assert ival[0] <= ival[1]
    # create an indexing mask ([true, true, false, ...])
    mask = np.logical_and(ival[0] <= epo.t, epo.t <= ival[1])
    # take all values from the epo except the ones not fitting the mask
    # and calculate the average along the sampling axis
    averages = np.average(epo.data[:, mask, :], axis=1)
    data = epo.data - averages[:, np.newaxis, :]
    return Epo(data, epo.fs, epo.channels, epo.markers, epo.classes, epo.class_names, epo.t[0])


def rectify_chanels(epo):
    """Calculate all samplewise absolute values.

    Parameters
    ----------
    epo : Epo

    Returns
    -------
    epo : Epo

    Examples
    --------

    >>> print np.average(epo.data)
    0.391987338917
    >>> epo = misc.rectify_chanels(epo)
    >>> print np.average(epo.data)
    22.40234266

    """
    data = np.abs(epo.data)
    return Epo(data, epo.fs, epo.channels, epo.markers, epo.classes, epo.class_names, epo.t[0])


def jumping_means(epo, ivals):
    """Calculate the jumping means.

    Parameters
    ----------
    epo : Epo
    ivals : array of [float, float]
        the intervals for which to calculate the means

    Returns
    -------

    """
    n_epos, n_samples, n_chans = epo.data.shape
    n_ivals = len(ivals)
    data = np.zeros((n_epos, n_ivals, n_chans))
    for i, [start, end] in enumerate(ivals):
        mask = np.logical_and(start <= epo.t, epo.t <= end)
        data[:, i, :] = np.mean(epo.data[:, mask, :], axis=1)
    t = np.mean(ivals, axis=1)
    # is this really an epo? what about:
    # - fs
    # - markers
    # - init of t_start
    epo = Epo(data, epo.fs, epo.channels, epo.markers, epo.classes, epo.class_names, 0)
    epo.t = t
    epo.markers = [[] for i in range(n_epos)]
    return epo
