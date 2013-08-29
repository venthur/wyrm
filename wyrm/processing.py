#!/usr/bin/env python

"""Processing toolbox methods.

This module contains the processing methods.

"""


from __future__ import division

import logging
import re

import numpy as np
import scipy as sp
from scipy import signal

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)



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
        included, the endpoint is not (like: ``[start, end)``).  To get
        200ms before the marker until 100ms after the marker do:
        ``[-200, 100]`` Only negative or positive values are possible
        (i.e. ``[-500, -100]``)
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


def band_pass(dat, low, high, timeaxis=-2):
    """Band pass filter the data.

    Parameters
    ----------
    dat : Date
    low, high : int
        the low, and high borders of the desired frequency band
    timeaxis : int, optional
        the axis in ``dat.data`` to filter

    Returns
    -------
    dat : Data
        the band pass filtered data

    """
    # band pass filter the data
    fs_n = dat.fs * 0.5
    #logger.debug('Calculating butter order...')
    #butter_ord, f_butter = signal.buttord(ws=[(low - .1) / fs_n, (high + .1) / fs_n],
    #                                      wp=[low / fs_n, high / fs_n],
    #                                      gpass=0.1,
    #                                      gstop=3.0
    #                                      )

    #logger.debug("order: {ord} fbutter: {fbutter} low: {low} high: {high}".format(**{'ord': butter_ord,
    #                                                      'fbutter': f_butter,
    #                                                      'low': low / fs_n,
    #                                                      'high': high / fs_n}))
    logger.warning('Using fixed order for butterworth filter.')
    butter_ord = 4
    b, a = signal.butter(butter_ord, [low / fs_n, high / fs_n], btype='band')
    data = signal.lfilter(b, a, dat.data, axis=timeaxis)
    return dat.copy(data=data)


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
    invert : Boolean, optional
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

    >>> dat.axes[0]
    [0, 0, 1, 2, 2]
    >>> dat = select_epochs(dat, [0, 1, 2])
    >>> dat.axes[0]
    [0, 0, 1]

    Remove the fourth epoch

    >>> dat.axes[0]
    [0, 0, 1, 2, 2]
    >>> dat = select_epochs(dat, [3], invert=True)
    >>> dat.axes[0]
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

    This method just calls :func:`select_epochs` with the ``inverse``
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


def select_classes(dat, indices, invert=False, classaxis=0):
    """Select classes from an epoched data object.

    This method selects the classes with the specified indices.

    Parameters
    ----------
    dat : Data
        epoched Data object
    indices : array of ints
        The indices of the classes to select.
    invert : Boolean, optional
        if true keep all classes except the ones defined by ``indices``.
    classaxis : int, optional
        the axis along which the classes are selected

    Returns
    -------
    dat : Data
        a copy of the epoched data with only the selected classes
        included.

    Raises
    ------
    AssertionError : if ``dat`` has no ``.class_names`` attribute.

    See Also
    --------
    remove_classes

    Examples
    --------

    Get the classes 1 and 2.

    >>> dat.axes[0]
    [0, 0, 1, 2, 2]
    >>> dat = select_classes(dat, [1, 2])
    >>> dat.axes[0]
    [1, 2, 2]

    Remove class 2

    >>> dat.axes[0]
    [0, 0, 1, 2, 2]
    >>> dat = select_classes(dat, [2], invert=True)
    >>> dat.axes[0]
    [0, 0, 1]

    """
    assert hasattr(dat, 'class_names')
    mask = np.array([False for i in range(dat.data.shape[classaxis])])
    for idx, val in enumerate(dat.axes[classaxis]):
        if val in indices:
            mask[idx] = True
    if invert:
        mask = ~mask
    data = dat.data.compress(mask, classaxis)
    axes = dat.axes[:]
    axes[classaxis] = dat.axes[classaxis].compress(mask)
    return dat.copy(data=data, axes=axes)


def remove_classes(*args, **kwargs):
    """Remove classes from an epoched Data object.

    This method just calls :func:`select_epochs` with the ``inverse``
    parameter set to ``True``.

    Returns
    -------
    dat : Data
        copy of Data object with the classes removed

    See Also
    --------
    select_classes

    """
    return select_classes(*args, invert=True, **kwargs)


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
    AssertionError : if ``dat.data.shape[timeaxis] != len(dat.axes[timeaxis])``

    """
    assert hasattr(dat, 'fs')
    assert dat.data.shape[timeaxis] == len(dat.axes[timeaxis])
    assert dat.fs % freq == 0
    factor = int(dat.fs / freq)
    idxmask = np.arange(dat.data.shape[timeaxis], step=factor)
    data = dat.data.take(idxmask, timeaxis)
    axes = dat.axes[:]
    axes[timeaxis] =  axes[timeaxis].take(idxmask)
    return dat.copy(data=data, axes=axes, fs=freq)


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


def create_feature_vectors(dat, classaxis=0):
    """Create feature vectors from epoched data.

    This method flattens a ``Data`` objects down to 2 dimensions: the
    first one for the classes and the second for the feature vectors.
    All surplus dimensions of the ``dat`` argument are clashed into the
    appropriate class.

    Parameters
    ----------
    dat : Data
    classaxis : int, optional
        the index of the class axis

    Returns
    -------
    dat : Data
        a copy of ``dat`` with reshaped to 2 dimensions and with the
        classaxis moved to dimension 0

    Examples
    --------

    >>> dat.shape
    (300, 2, 64)
    >>> dat = create_feature_vectors(dat)
    >>> dat.shape
    (300, 128)

    """
    if classaxis != 0:
        dat = swapaxes(dat, 0, classaxis)
    data = dat.data.reshape(dat.data.shape[classaxis], -1)
    axes = dat.axes[:]
    axes[-1] = np.arange(data.shape[-1])
    names = dat.names[:]
    names[-1] = 'feature vector'
    units = dat.units[:]
    units[-1] = 'dl'
    return dat.copy(data=data, axes=axes, names=names, units=units)


def calculate_signed_r_square(dat, classaxis=0):
    """Calculate the signed r**2 values.

    This method calculates the signed r**2 values over the epochs of the
    ``dat``.

    Parameters
    ----------
    dat : Data
        epoched data
    classaxis : int, optional
        the axis to be treatet as the classaxis

    Returns
    -------
    signed_r_square : ndarray
        the signed r**2 values, signed_r_square has one axis less than
        the ``dat`` parameter, the ``classaxis`` has been removed

    Examples
    --------

    >>> dat.data.shape
    (400, 100, 64)
    >>> r = calculate_signed_r_square(dat)
    >>> r.shape
    (100, 64)

    """
    # TODO: explain the algorithm in the docstring and add a reference
    # to a paper.
    # select class 0 and 1
    # TODO: make class 0, 1 variables
    fv1 = select_classes(dat, [0], classaxis=classaxis).data
    fv2 = select_classes(dat, [1], classaxis=classaxis).data
    # number of epochs per class
    l1 = fv1.shape[classaxis]
    l2 = fv2.shape[classaxis]
    # calculate r-value (Benjamin approved!)
    a = (fv1.mean(axis=classaxis) - fv2.mean(axis=classaxis)) * np.sqrt(l1 * l2)
    b = dat.data.std(axis=classaxis) * (l1 + l2)
    r = a / b
    # return signed r**2
    s = np.sign(r)
    return s * r * r


def logarithm(dat):
    """Computes the element wise natural logarithm of ``dat.data``

    Calling this method is equivalent to calling

    >>> dat.copy(data=np.log(dat.data))

    Parameters
    ----------
    dat : Data

    Returns
    -------
    dat : Data
        a copy of ``dat`` with the element wise natural logarithms of
        the values in ``.data``

    """
    data = np.log(dat.data)
    return dat.copy(data=data)


def variance(dat, timeaxis=-2):
    """Compute the variance along the ``timeaxis`` of ``dat``.

    Parameters
    ----------
    dat : Data

    Returns
    -------
    dat : Data
        copy of ``dat`` with with the variance along the ``timeaxis``
        removed and ``timeaxis`` removed.

    Examples
    --------

    >>> epo.names
    ['class', 'time', 'channel']
    >>> var = variance(cnt)
    >>> var.names
    ['class', 'channel']

    """
    data = np.var(dat.data, axis=timeaxis)
    axes = dat.axes[:].pop(timeaxis)
    names = dat.names[:].pop(timeaxis)
    units = dat.units[:].pop(timeaxis)
    return dat.copy(data=data, axes=axes, names=names, units=units)

