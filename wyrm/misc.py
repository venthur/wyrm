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
import copy

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


class Data(object):
    """Not documented yet :(

    Parameters
    ----------
    data : ndarray
    axes : nlist of 1darrays
    names : nlist of strings
    units : nlist of strings

    Attributes
    ----------
    data : ndarray
    axes : nlist of 1darrays
    names : nlist of strings
    units : nlist of strings

    """
    def __init__(self, data, axes, names, units):
        assert data.ndim == len(axes) == len(names) == len(units)
        for i in range(data.ndim):
            if data.shape[i] != len(axes[i]):
                raise AssertionError("Axis '%s' (%i) not as long as corresponding axis in 'data' (%i)" % (names[i], len(axes[i]), data.shape[i]))
        self.data = data
        self.axes = [np.array(i) for i in axes]
        self.names = names
        self.units = units

    def __eq__(self, other):
        """Test for equality.

        Don't trust this method it only checks for known attributes and
        assumes equality if those are equal.

        Parameters
        ----------
        other : Data

        Returns
        -------
        equal : Boolean
            True if ``self`` and ``other`` are equal, False if not.

        """
        if (sorted(self.__dict__.keys()) == sorted(other.__dict__.keys()) and
            np.array_equal(self.data, other.data) and
            len(self.axes) == len(other.axes) and
            all([self.axes[i].shape == other.axes[i].shape for i in range(len(self.axes))]) and
            all([(self.axes[i] == other.axes[i]).all() for i in range(len(self.axes))]) and
            self.names == other.names and
            self.units == other.units
           ):
            return True
        return False

    def copy(self, **kwargs):
        """Return a memory efficient deep copy of ``self``.

        It first creates a shallow copy of ``self``, sets the attributes
        in ``kwargs`` if necessary and returns a deep copy of the
        resulting object.

        Parameters
        ----------
        kwargs : dict, optional
            if provided ``copy`` will try to overwrite the name, value
            pairs after the shallow- and before the deep copy. If no
            ``kwargs`` are provided, it will just return the deep copy.

        Returns
        -------
        dat : Data
            a deep copy of ``self``.

        Examples
        --------
        >>> # perform an ordinary deep copy of dat
        >>> dat2 = dat.copy()
        >>> # perform a deep copy but overwrite .axes first
        >>> dat.axes
        ['time', 'channels']
        >>> dat3 = dat.copy(axes=['foo'], ['bar'])
        >>> dat3.axes
        ['foo', 'bar']
        >>> dat.axes
        ['time', 'channel']

        """
        obj = copy.copy(self)
        for name, value in kwargs.items():
            setattr(obj, name, value)
        return copy.deepcopy(obj)


def swapaxes(dat, ax1, ax2):
    """Swap axes of a Data object.

    This method swaps two axes of a Data object by swapping the
    appropriate ``.data``, ``.names``, ``.units``, and ``.axes``.

    Parameters
    ----------
    dat : Data
    ax1, ax2 : int
        the indices of the axes to swap

    Returns
    -------
    dat : Data
        a copy of ``dat`` with the appropriate axes swapped.

    Examples
    --------
    >>> dat.names
    ['time', 'channels']
    >>> dat = swapaxes(dat, 0, 1)
    >>> dat.names
    ['channels', 'time']

    See Also
    --------
    numpy.swapaxes

    """
    data = dat.data.swapaxes(ax1, ax2)
    axes = dat.axes[:]
    axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
    units = dat.units[:]
    units[ax1], units[ax2] = units[ax2], units[ax1]
    names = dat.names[:]
    names[ax1], names[ax2] = names[ax2], names[ax1]
    return dat.copy(data=data, axes=axes, units=units, names=names)


def select_channels(dat, regexp_list, invert=False, chanaxis=-1):
    """Select channels from data.

    The matching is case-insensitive and locale-aware (as in
    ``re.IGNORECASE`` and ``re.LOCALE``). The regular expression always
    has to match the whole channel name string

    Parameters
    ----------
    dat : Data
    regexp_list : list of regular expressions
        The regular expressions provided, are used directly by Python's
        :mod:`re` module, so all regular expressions which are understood
        by this module are allowed.

        Internally the :func:`re.match` method is used, additionally to
        check for a match (which also matches substrings), it is also
        checked if the whole string matched the pattern.
    invert : Boolean, optional
        If True the selection is inverted. Instead of selecting specific
        channels, you are removing the channels. (default: False)
    chanaxis : int, optional
        the index of the channel axis in ``dat`` (default: -1)

    Returns
    -------
    dat : Data
        A copy of ``dat`` with the channels, matched by the list of
        regular expressions.

    Examples
    --------
    Select all channels Matching 'af.*' or 'fc.*'

    >>> dat_new = select_channels(dat, ['af.*', 'fc.*'])

    Remove all channels Matching 'emg.*' or 'eog.*'

    >>> dat_new = select_channels(dat, ['emg.*', 'eog.*'], invert=True)

    Even if you only provide one Regular expression, it has to be in an
    array:

    >>> dat_new = select_channels(dat, ['af.*'])

    See Also
    --------
    remove_channels : Remove Channels
    re : Python's Regular Expression module for more information about
        regular expressions.

    """
    # TODO: make it work with epos
    chan_mask = np.array([False for i in range(len(dat.axes[chanaxis]))])
    for c_idx, c in enumerate(dat.axes[chanaxis]):
        for regexp in regexp_list:
            m = re.match(regexp, c, re.IGNORECASE | re.LOCALE)
            if m and m.group() == c:
                chan_mask[c_idx] = True
                # no need to look any further for matches for this channel
                break
    if invert:
        chan_mask = ~chan_mask
    data = dat.data.compress(chan_mask, chanaxis)
    channels = dat.axes[chanaxis][chan_mask]
    axes = dat.axes[:]
    axes[chanaxis] = channels
    return dat.copy(data=data, axes=axes)


def remove_channels(*args, **kwargs):
    """Remove channels from data.

    This method just calls :func:`select_channels` with the same
    parameters and the ``invert`` parameter set to ``True``.

    Returns
    -------
    dat : Data
        A copy of the dat with the channels removed.

    See Also
    --------
    select_channels : Select Channels

    """
    return select_channels(*args, invert=True, **kwargs)


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


def segment_dat(dat, marker_def, ival, timeaxis=-2):
    """Convert a continuous data object to an epoched one.

    Given a continuous data object, a definition of classes, and an
    interval, this method looks for markers as defined in ``marker_def``
    and slices the dat according to the time interval given with
    ``ival`` along the ``timeaxis``. The returned ``dat`` object stores
    those slices and the class each slice belongs to.


    Parameters
    ----------
    dat : Data
    marker_def : dict
        The keys are class names, the values are lists of markers
    ival : [int, int]
        The interval in milliseconds to cut around the markers. I.e. to
        get the interval starting with the marker plus the remaining
        100ms define the interval like [0, 100]. The start point is
        included, the endpoint is not (like: ``[start, end)``).

        To get 200ms before the marker until 100ms after the marker do:
        ``[-200, 100]``

        Only negative or positive values are possible (i.e. ``[-500,
        -100]``)
    timeaxis : int, optional
        the axis along which the segmentation will take place

    Returns
    -------
    dat : Data
        a copy of the resulting epoched data.

    Raises
    ------
    AssertionError :
        if ``dat`` has not ``.fs`` or ``.markers`` attribute or if
        ``ival[0] > ival[1]``.

    Examples
    --------
    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # Epoch the data -500ms and +700ms around the markers defined in
    >>> # md
    >>> epo = segment_dat(cnt, md, [-500, 700])

    """
    assert hasattr(dat, 'fs')
    assert hasattr(dat, 'markers')
    assert ival[0] <= ival[1]
    data = []
    classes = []
    class_names = sorted(marker_def.keys())
    for t, m in dat.markers:
        for class_idx, classname in enumerate(class_names):
            if m in marker_def[classname]:
                mask = (t+ival[0] <= dat.axes[timeaxis]) & (dat.axes[timeaxis] < t+ival[1])
                d = dat.data.compress(mask, timeaxis)
                d = np.expand_dims(d, axis=0)
                data.append(d)
                classes.append(class_idx)
    data = np.concatenate(data, axis=0)
    axes = dat.axes[:]
    time = np.linspace(ival[0], ival[1], (ival[1] - ival[0]) / 1000 * dat.fs, endpoint=False)
    axes[timeaxis] = time
    classes = np.array(classes)
    axes.insert(0, classes)
    names = dat.names[:]
    names.insert(0, 'class')
    units = dat.units[:]
    units.insert(0, '#')
    return dat.copy(data=data, axes=axes, names=names, units=units, class_names=class_names)


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


def select_ival(dat, ival, timeaxis=-2):
    """Select interval from data.

    This method selects the time segment(s) defined by ``ival``.

    Parameters
    ----------
    dat : Data
    ival : list of two floats
        Start and end in milliseconds. Start is included end is excluded
        (like ``[stard, end)``]
    timeaxis : int, optional
        the axis along which the intervals are selected

    Returns
    -------
    dat : Data
        a copy of ``dat`` with the selected time intervals.

    Raises
    ------
    AssertionError
        if the given interval does not fit into ``dat.axes[timeaxis]``
        or ``ival[0] > ival[1]``.

    Examples
    --------

    Select the first 200ms of the epoched data:

    >>> dat.fs
    100.
    >>> dat2 = select_ival(dat, [0, 200])
    >>> print dat2.t[0], dat2.t[-1]
    0. 199.

    """
    assert dat.axes[timeaxis][0] <= ival[0] <= ival[1]
    mask = (ival[0] <= dat.axes[timeaxis]) & (dat.axes[timeaxis] < ival[1])
    data = dat.data.compress(mask, timeaxis)
    axes = dat.axes[:]
    axes[timeaxis] = dat.axes[timeaxis].compress(mask)
    return dat.copy(data=data, axes=axes)


def select_epochs(dat, indices, invert=False, classaxis=0):
    """Select epochs from an epoched data object.

    This method selects the epochs with the specified indices.

    Parameters
    ----------
    dat : Data
        epoched Data object with an ``.class_names`` attribute
    indices : array of ints
        The indices of the elements to select.
    invert : Boolean
        if true keep all elements except the ones defined by ``indices``.
    classaxis : int, optional
        the axis along which the epochs are selected

    Returns
    -------
    dat : Data
        a copy of the epoched data with only the selected epochs included.

    Raises
    ------
    AssertionError : if ``dat`` has no ``.class_names`` attribute.

    See Also
    --------
    remove_epochs

    Examples
    --------

    Get the first three epochs.

    >>> dat.classes
    [0, 0, 1, 2, 2]
    >>> dat = select_epochs(dat, [0, 1, 2])
    >>> dat.classes
    [0, 0, 1]

    Remove the fourth epoch

    >>> dat.classes
    [0, 0, 1, 2, 2]
    >>> dat = select_epochs(dat, [3], invert=True)
    >>> dat.classes
    [0, 0, 1, 2]

    """
    assert hasattr(dat, 'class_names')
    mask = np.array([False for i in range(dat.data.shape[classaxis])])
    for i in indices:
        mask[i] = True
    if invert:
        mask = ~mask
    data = dat.data.compress(mask, classaxis)
    axes = dat.axes[:]
    axes[classaxis] = dat.axes[classaxis].compress(mask)
    return dat.copy(data=data, axes=axes)


def remove_epochs(*args, **kwargs):
    """Remove epochs from an epoched Data object.

    This method just calls :meth:`select_epochs` with the ``inverse``
    paramerter set to ``True``.

    Returns
    -------
    dat : Data
        epoched Data object with the epochs removed

    See Also
    --------
    select_epochs

    """
    return select_epochs(*args, invert=True, **kwargs)


def subsample(dat, freq, timeaxis=-2):
    """Subsample the data to ``freq`` Hz.

    This method subsamples data along ``timeaxis`` by taking every ``n``
    th element starting with the first one and ``n`` being ``dat.fs /
    freq``. Please note that ``freq`` must be a whole number divisor of
    ``dat.fs``.

    Note that this method does not low-pass filter the data before
    sub-sampling.

    Parameters
    ----------
    dat : Data
        Data object with ``.fs`` attribute
    freq : float
        the target frequency in Hz
    timeaxis : int, optional
        the axis along which to subsample

    Returns
    -------
    dat : Data
        copy of ``dat`` with subsampled frequency

    See Also
    --------
    band_pass

    Examples
    --------

    Load some EEG data with 1kHz, bandpass filter it and downsample it
    to 100Hz.

    >>> dat = load_brain_vision_data('some/path')
    >>> dat.fs
    1000.0
    >>> dat = band_pass(dat, 8, 40)
    >>> dat = subsample(dat, 100)
    >>> dat.fs
    100.0

    Raises
    ------
    AssertionError : if ``freq`` is not a whole number divisor of ``dat.fs``
    AssertionError : if ``dat`` has no ``.fs`` attribute
    AssertionError : if ``dat.data.shape[timeaxis] != len(dat.axes[timexis])``

    """
    assert hasattr(dat, 'fs')
    assert dat.data.shape[timeaxis] == len(dat.axes[timeaxis])
    assert dat.fs % freq == 0
    factor = int(dat.fs / freq)
    idxmask = np.arange(dat.data.shape[timeaxis], step=factor)
    data = dat.data.take(idxmask, timeaxis)
    axes = dat.axes[:]
    axes[timeaxis] =  axes[timeaxis].take(idxmask)
    markers = map(lambda x: [int(x[0] / factor), x[1]], dat.markers)
    return dat.copy(data=data, axes=axes, fs=freq, markers=markers)


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


def calculate_classwise_average(dat, classaxis=0):
    """Calculate the classwise average.

    This method calculates the average continuous per class for all
    classes defined in the ``dat``. In other words, if you have two
    different classes, with many continuous data per class, this method
    will calculate the average time course for each class and channel.

    Parameters
    ----------
    dat : Data
        an epoched Data object with a ``.class_names`` attribute.
    classaxis : int, optional
        the axis along which to calculate the average

    Returns
    -------
    dat : Data
        copy of ``dat`` a witht the ``classaxis`` dimension reduced to
        the number of different classes.

    Raises
    ------
    AssertionError : if the ``dat`` has no ``.class_names`` attribute.

    Examples
    --------
    Split existing continuous data into two classes and calculate the
    average for each class.

    >>> mrk_def = {'std': ['S %2i' % i for i in range(2, 7)],
    ...            'dev': ['S %2i' % i for i in range(12, 17)]
    ...           }
    >>> epo = misc.segment_dat(cnt, mrk_def, [0, 660])
    >>> avg_epo = calculate_classwise_average(epo)
    >>> plot(avg_epo.data[0])
    >>> plot(avg_epo.data[1])

    """
    assert hasattr(dat, 'class_names')
    classes = []
    data = []
    for i, classname in enumerate(dat.class_names):
        avg = np.average(dat.data[dat.axes[classaxis] == i], axis=classaxis)
        data.append(avg)
        classes.append(i)
    data = np.array(data)
    classes = np.array(classes)
    axes = dat.axes[:]
    axes[classaxis] = classes
    return dat.copy(data=data, axes=axes)


def correct_for_baseline(dat, ival, timeaxis=-2):
    """Subtract the baseline.

    For each epoch and channel in the given dat, this method calculates
    the average value for the given interval and subtracts this value
    from the channel data within this epoch and channel.

    This method generalizes to dats with more than 3 dimensions.

    Parameters
    ----------
    dat : Dat
    ival : array of two floats
        the start and stop borders in milli seconds. the left border is
        included, the right border is not: ``[start, stop)``.
        ``ival[0]`` must fit into ``dat.axes[timeaxis]`` and
        ``ival[0] <= ival[1]``.
    timeaxis : int, optional
        the axis along which to correct for the baseline

    Returns
    -------
    dat : Dat
        a copy of ``dat`` with the averages of the intervals subtracted.

    Examples
    --------

    Remove the baselines for the interval ``[100, 0)``

    >>> dat = correct_for_baseline(dat, [-100, 0])

    Notes
    -----
    The Algorithm calculates the average(s) along the ``timeaxis`` within
    the given interval. The resulting array has one dimension less
    than the original one (the elements on ``timeaxis`` where reduced).

    The resulting avgarray is then subtracted from the original data. To
    match the shape, a new axis is created on ``timeaxis`` of avgarray.
    And the shapes are then matched via numpy's broadcasting.

    Raises
    ------
    AssertionError
        If the left border of ``ival`` is outside of
        ``dat.axes[timeaxis]`` or if ``ival[1] < ival[0]``.

    See Also
    --------
    numpy.average, numpy.expand_dims

    """
    # check if ival fits into dat.ival
    # we can't make any assumptions about ival[1] and the last element
    # of the timeaxis since ival[1] is expected to be bigger as the
    # interval is not including the last element of ival[1]
    assert dat.axes[timeaxis][0] <= ival[0] <= ival[1]
    mask = (ival[0] <= dat.axes[timeaxis]) & (dat.axes[timeaxis] < ival[1])
    # take all values from the dat except the ones not fitting the mask
    # and calculate the average along the sampling axis
    averages = np.average(dat.data.compress(mask, timeaxis), axis=timeaxis)
    data = dat.data - np.expand_dims(averages, timeaxis)
    return dat.copy(data=data)


def rectify_channels(dat):
    """Calculate the absolute values in ``dat.data``.

    Parameters
    ----------
    dat : Data

    Returns
    -------
    dat : Data
        a copy of ``dat`` with all values absolute in ``.data``

    Examples
    --------

    >>> print np.average(dat.data)
    0.391987338917
    >>> dat = rectify_channels(dat)
    >>> print np.average(dat.data)
    22.40234266

    """
    return dat.copy(data=np.abs(dat.data))


def jumping_means(dat, ivals, timeaxis=-2):
    """Calculate the jumping means.

    Parameters
    ----------
    dat : Data
    ivals : array of [float, float]
        the intervals for which to calculate the means. Start is
        included end is not (like ``[start, end)``).
    timeaxis : int, optional
        the axis along which to calculate the jumping means

    Returns
    -------
    dat : Data
        copy of ``dat`` with the jumping means along the ``timeaxis``.
        ``dat.name[timeaxis]`` and ``dat.axes[timeaxis]`` Are modified
        too to reflect the intervals used for the data points.

    """
    means = []
    time = []
    for i, [start, end] in enumerate(ivals):
        mask = (start <= dat.axes[timeaxis]) & (dat.axes[timeaxis] < end)
        mean = np.mean(dat.data.compress(mask, timeaxis), axis=timeaxis)
        mean = np.expand_dims(mean, timeaxis)
        means.append(mean)
        time.append('[%i, %i)' % (start, end))
    means = np.concatenate(means, axis=timeaxis)
    names = dat.names[:]
    names[timeaxis] = 'time interval'
    axes = dat.axes[:]
    axes[timeaxis] = np.array(time)
    return dat.copy(data=means, names=names, axes=axes)

