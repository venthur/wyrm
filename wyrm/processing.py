#!/usr/bin/env python
# coding=utf8

"""Processing toolbox methods.

This module contains the processing methods.

"""


from __future__ import division

import functools
import logging
import re

import numpy as np
import scipy as sp
from scipy import signal
from sklearn.covariance import LedoitWolf as LW

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


# the angles here are given in (90 / 4)th degrees - so multiply it with
# (90 / 4) to get the actual angles
CHANNEL_10_20 = (
    ('Fpz', (0.0, 4.0)),
    ('Fp1', (-4.0, 3.5)),
    ('AFp1', (-1.5, 3.5)),
    ('AFp2', (1.5, 3.5)),
    ('Fp2', (4.0, 3.5)),
    ('AF7', (-4.0, 3.0)),
    ('AF5', (-3.0, 3.0)),
    ('AF3', (-2.0, 3.0)),
    ('AFz', (0.0, 3.0)),
    ('AF4', (2.0, 3.0)),
    ('AF6', (3.0, 3.0)),
    ('AF8', (4.0, 3.0)),
    ('FAF5', (-2.5, 2.5)),
    ('FAF1', (-0.65, 2.5)),
    ('FAF2', (0.65, 2.5)),
    ('FAF6', (2.5, 2.5)),
    ('F9', (-5.0, 2.0)),
    ('F7', (-4.0, 2.0)),
    ('F5', (-3.0, 2.0)),
    ('F3', (-2.0, 2.0)),
    ('F1', (-1.0, 2.0)),
    ('Fz', (0.0, 2.0)),
    ('F2', (1.0, 2.0)),
    ('F4', (2.0, 2.0)),
    ('F6', (3.0, 2.0)),
    ('F8', (4.0, 2.0)),
    ('F10', (5.0, 2.0)),
    ('FFC9', (-4.5, 1.5)),
    ('FFC7', (-3.5, 1.5)),
    ('FFC5', (-2.5, 1.5)),
    ('FFC3', (-1.5, 1.5)),
    ('FFC1', (-0.5, 1.5)),
    ('FFC2', (0.5, 1.5)),
    ('FFC4', (1.5, 1.5)),
    ('FFC6', (2.5, 1.5)),
    ('FFC8', (3.5, 1.5)),
    ('FFC10', (4.5, 1.5)),
    ('FT9', (-5.0, 1.0)),
    ('FT7', (-4.0, 1.0)),
    ('FC5', (-3.0, 1.0)),
    ('FC3', (-2.0, 1.0)),
    ('FC1', (-1.0, 1.0)),
    ('FCz', (0.0, 1.0)),
    ('FC2', (1.0, 1.0)),
    ('FC4', (2.0, 1.0)),
    ('FC6', (3.0, 1.0)),
    ('FT8', (4.0, 1.0)),
    ('FT10', (5.0, 1.0)),
    ('CFC9', (-4.5, 0.5)),
    ('CFC7', (-3.5, 0.5)),
    ('CFC5', (-2.5, 0.5)),
    ('CFC3', (-1.5, 0.5)),
    ('CFC1', (-0.5, 0.5)),
    ('CFC2', (0.5, 0.5)),
    ('CFC4', (1.5, 0.5)),
    ('CFC6', (2.5, 0.5)),
    ('CFC8', (3.5, 0.5)),
    ('CFC10', (4.5, 0.5)),
    ('T9', (-5.0, 0.0)),
    ('T7', (-4.0, 0.0)),
    ('C5', (-3.0, 0.0)),
    ('C3', (-2.0, 0.0)),
    ('C1', (-1.0, 0.0)),
    ('Cz', (0.0, 0.0)),
    ('C2', (1.0, 0.0)),
    ('C4', (2.0, 0.0)),
    ('C6', (3.0, 0.0)),
    ('T8', (4.0, 0.0)),
    ('T10', (5.0, 0.0)),
    ('A1', (-5.0, -0.5)),
    ('CCP7', (-3.5, -0.5)),
    ('CCP5', (-2.5, -0.5)),
    ('CCP3', (-1.5, -0.5)),
    ('CCP1', (-0.5, -0.5)),
    ('CCP2', (0.5, -0.5)),
    ('CCP4', (1.5, -0.5)),
    ('CCP6', (2.5, -0.5)),
    ('CCP8', (3.5, -0.5)),
    ('A2', (5.0, -0.5)),
    ('TP9', (-5.0, -1.0)),
    ('TP7', (-4.0, -1.0)),
    ('CP5', (-3.0, -1.0)),
    ('CP3', (-2.0, -1.0)),
    ('CP1', (-1.0, -1.0)),
    ('CPz', (0.0, -1.0)),
    ('CP2', (1.0, -1.0)),
    ('CP4', (2.0, -1.0)),
    ('CP6', (3.0, -1.0)),
    ('TP8', (4.0, -1.0)),
    ('TP10', (5.0, -1.0)),
    ('PCP9', (-4.5, -1.5)),
    ('PCP7', (-3.5, -1.5)),
    ('PCP5', (-2.5, -1.5)),
    ('PCP3', (-1.5, -1.5)),
    ('PCP1', (-0.5, -1.5)),
    ('PCP2', (0.5, -1.5)),
    ('PCP4', (1.5, -1.5)),
    ('PCP6', (2.5, -1.5)),
    ('PCP8', (3.5, -1.5)),
    ('PCP10', (4.5, -1.5)),
    ('P9', (-5.0, -2.0)),
    ('P7', (-4.0, -2.0)),
    ('P5', (-3.0, -2.0)),
    ('P3', (-2.0, -2.0)),
    ('P1', (-1.0, -2.0)),
    ('Pz', (0.0, -2.0)),
    ('P2', (1.0, -2.0)),
    ('P4', (2.0, -2.0)),
    ('P6', (3.0, -2.0)),
    ('P8', (4.0, -2.0)),
    ('P10', (5.0, -2.0)),
    ('PPO7', (-4.5, -2.5)),
    ('PPO5', (-3.0, -2.5)),
    ('PPO3', (-2.0, -2.5)),
    ('PPO1', (-0.65, -2.5)),
    ('PPO2', (0.65, -2.5)),
    ('PPO4', (2.0, -2.5)),
    ('PPO6', (3.0, -2.5)),
    ('PPO8', (4.5, -2.5)),
    ('PO9', (-5.5, -2.6)),
    ('PO7', (-4.0, -3)),
    ('PO5', (-3.0, -3)),
    ('PO3', (-2.0, -3)),
    ('PO1', (-1.0, -3)),
    ('POz', (0.0, -3)),
    ('PO2', (1.0, -3)),
    ('PO4', (2.0, -3)),
    ('PO6', (3.0, -3)),
    ('PO8', (4.0, -3)),
    ('PO10', (5.5, -2.6)),
    ('OPO1', (-1.5, -3.5)),
    ('OPO2', (1.5, -3.5)),
    ('O9', (-6.5, -3.5)),
    ('O1', (-4.0, -3.5)),
    ('O2', (4.0, -3.5)),
    ('O10', (6.5, -3.5)),
    ('Oz', (0.0, -4.0)),
    ('OI1', (1.5, -4.5)),
    ('OI2', (-1.5, -4.5)),
    ('I1', (1.0, -5)),
    ('Iz', (0.0, -5)),
    ('I2', (-1, -5))
)


def lda_train(fv, shrink=False):
    """Train the LDA classifier.

    Parameters
    ----------
    fv : ``Data`` object
        the feature vector must have 2 dimensional data, the first
        dimension being the class axis. The unique class labels must be
        0 and 1 otherwise a ``ValueError`` will be raised.
    shrink : Boolean, optional
        use shrinkage

    Returns
    -------
    w : 1d array
    b : float

    Raises
    ------
    ValueError : if the class labels are not exactly 0s and 1s

    Examples
    --------

    >>> clf = lda_train(fv_train)
    >>> out = lda_apply(fv_test, clf)

    See Also
    --------
    lda_apply

    """
    x = fv.data
    y = fv.axes[0]
    # TODO: this code should be trivially fixed to allow for two unique
    # values instead of 0, 1 hardcoded
    if not np.all(np.unique(y) == [0, 1]):
        raise ValueError('Unique class labels are {labels}, should be [0, 1]'.format(labels=np.unique(y)))
    mu1 = np.mean(x[y == 0], axis=0)
    mu2 = np.mean(x[y == 1], axis=0)
    # x' = x - m
    m = np.empty(x.shape)
    m[y == 0] = mu1
    m[y == 1] = mu2
    x2 = x - m
    # w = cov(x)^-1(mu2 - mu1)
    if shrink:
        covm = LW().fit(x2).covariance_
    else:
        covm = np.cov(x2.T)
    w = np.dot(np.linalg.pinv(covm), (mu2 - mu1))
    # b = 1/2 x'(mu1 + mu2)
    b = -0.5 * np.dot(w.T, (mu1 + mu2))
    return w, b


def lda_apply(fv, clf):
    """Apply feature vector to LDA classifier.

    Parameters
    ----------
    fv : ``Data`` object
        the feature vector must have a 2 dimensional data, the first
        dimension being the class axis.
    clf : (1d array, float)

    Returns
    -------

    out : 1d array
        The projection of the data on the hyperplane.

    Examples
    --------

    >>> clf = lda_train(fv_train)
    >>> out = lda_apply(fv_test, clf)


    See Also
    --------
    lda_train

    """
    x = fv.data
    w, b = clf
    return np.dot(x, w) + b


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


def sort_channels(dat, chanaxis=-1):
    """Sort channels.

    This method sorts the channels in the ``dat`` according to the 10-20
    system, from frontal to occipital and within the rows from left to
    right. The method uses the ``CHANNEL_10_20`` list and relies on the
    elements in that list to be sorted correctly. This method will put
    unknown channel names to the back of the resulting list.

    The channel matching is case agnostic.

    Parameters
    ----------
    dat : ``Data`` object
    chanaxis : int, optional
        the index of the channel axis in ``dat``

    Returns
    -------
    dat : ``Data`` object
        a copy of the ``dat`` parameter with the channels and data
        sorted.

    Examples
    --------

    >>> dat.axes[-1]
    array(['PPO4' 'CP4' 'PCP1' 'F5' 'C3' 'C4' 'O1' 'PPO2' 'FFC2' 'FAF5'
        'PO1' 'TP10' 'FAF1' 'FFC6' 'FFC1' 'PO10' 'O10' 'C1' 'Cz' 'F2'
        'CFC1' 'CCP2' 'F4' 'PO9' 'CFC6' 'TP7' 'FC6' 'AF8' 'Fz' 'AF4'
        'PCP9' 'F6' 'FT10' 'FAF6' 'PO5' 'O2' 'OPO2' 'AF5' 'C2' 'P4'
        'TP9' 'PCP7' 'FT8' 'A2' 'PO6' 'FC3' 'PPO1' 'CCP8' 'OPO1' 'AFp2'
        'OI2' 'OI1' 'FCz' 'CCP6' 'CCP1' 'CPz' 'POz' 'FFC3' 'FFC7' 'FC2'
        'F1' 'FT9' 'P2' 'P10' 'T9' 'FC1' 'C5' 'T7' 'CFC4' 'P6' 'F8'
        'TP8' 'CFC5' 'PCP8' 'CFC9' 'AF7' 'FC5' 'I1' 'CFC8' 'FFC8' 'Oz'
        'Pz' 'PCP4' 'FAF2' 'PCP5' 'CP1' 'PCP3' 'P1' 'Iz' 'CCP5' 'PO2'
        'PCP2' 'PO4' 'Fpz' 'F7' 'PO8' 'AFz' 'F10' 'FFC10' 'CCP3' 'PPO8'
        'T10' 'AF6' 'F9' 'PPO5' 'CP6' 'I2' 'PPO7' 'FC4' 'CCP4' 'PO7'
        'A1' 'CP2' 'CFC3' 'T8' 'PPO3' 'Fp2' 'PCP6' 'AFp1' 'C6' 'FFC9'
        'FT7' 'AF3' 'Fp1' 'CFC10' 'CCP7' 'CFC7' 'PO3' 'P7' 'P9' 'FFC4'
        'P5' 'CFC2' 'F3' 'CP3' 'PPO6' 'P3' 'O9' 'PCP10' 'P8' 'CP5'
        'FFC5'], dtype='|S5')
    >>> dat = sort_channels(dat)
    >>> dat.axes[-1]
    array(['Fpz', 'Fp1', 'AFp1', 'AFp2', 'Fp2', 'AF7', 'AF5', 'AF3',
        'AFz', 'AF4', 'AF6', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6',
        'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        'F10', 'FFC9', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4',
        'FFC6', 'FFC8', 'FFC10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC9', 'CFC7',
        'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'CFC10',
        'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        'T10', 'A1', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4',
        'CCP6', 'CCP8', 'A2', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz',
        'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP9', 'PCP7', 'PCP5',
        'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8', 'PCP10', 'P9',
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
        'PPO7', 'PPO5', 'PPO3', 'PPO1', 'PPO2', 'PPO4', 'PPO6', 'PPO8',
        'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO6',
        'PO8', 'PO10', 'OPO1', 'OPO2', 'O9', 'O1', 'O2', 'O10', 'Oz',
        'OI1', 'OI2', 'I1', 'Iz', 'I2'], dtype='|S5')
    """
    dat = dat.copy()
    # create the reference list of channel names in lower case
    ref = [name.lower() for name, _ in CHANNEL_10_20]
    # convert our channel names to lower case
    channels = [chan.lower() for chan in dat.axes[chanaxis]]
    # re-order the channels according to our 10-20 list:
    # iterate through all channel names in our list, put the known ones
    # into the `found` list along with their index in our list and the
    # index in the reference list. put all unknown channels into the
    # `notfound` list
    found = []
    notfound = []
    for idx_our, val in enumerate(channels):
        try:
            idx_ref = ref.index(val)
            found.append((idx_ref, idx_our))
        except ValueError:
            # not in reference!
            notfound.append(idx_our)
    # sort the `found` list wrt the indices of the reference list
    found.sort()
    # remove the reference indices again
    found = [i for _, i in found]
    # append found and notfound
    indices = found + notfound
    # do the actual re-ordering according to the indices
    dat.data = dat.data.take(indices, chanaxis)
    dat.axes[chanaxis] = dat.axes[chanaxis][indices]
    return dat


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


def segment_dat(dat, marker_def, ival, newsamples=None, timeaxis=-2):
    """Convert a continuous data object to an epoched one.

    Given a continuous data object, a definition of classes, and an
    interval, this method looks for markers as defined in ``marker_def``
    and slices the dat according to the time interval given with
    ``ival`` along the ``timeaxis``. The returned ``dat`` object stores
    those slices and the class each slice belongs to.

    Epochs that are too close to the borders and thus too short are
    ignored.

    If the segmentation does not result in any epochs (i.e. the markers
    in ``marker_def`` could not be found in ``dat``, the resulting
    dat.data will be an empty array.

    This method is also suitable for **online processing**, please read
    the documentation for the ``newsamples`` parameter and have a look
    at the Examples below.

    Parameters
    ----------
    dat : Data
        the data object to be segmented
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
    newsamples : int, optional
        consider the last ``newsamples`` samples as new data and only
        return epochs which are possible with the old **and** the new
        data (i.e. don't include epochs which where possible without the
        new data).

        If this parameter is ``None`` (default) ``segment_dat`` will
        always process the whole ``dat``, this is what you want for
        offline experiments where you process the whole data from a file
        at once. In online experiments however one usually gets the data
        incrementally, stores it in a ringbuffer to get the last n
        milliseconds. Consequently ``segment_dat`` gets overlapping data
        in each iteration (the amount of overlap is exactly the data -
        the new samples. To make sure each epoch appears only once
        within all iterations, ``segment_dat`` needs to know the number
        of new samples.


    timeaxis : int, optional
        the axis along which the segmentation will take place

    Returns
    -------
    dat : Data
        a copy of the resulting epoched data.

    Raises
    ------
    AssertionError
        * if ``dat`` has not ``.fs`` or ``.markers`` attribute or if
          ``ival[0] > ival[1]``.
        * if ``newsamples`` is not ``None`` or positive

    Examples
    --------

    Offline Experiment

    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # Epoch the data -500ms and +700ms around the markers defined in
    >>> # md
    >>> epo = segment_dat(cnt, md, [-500, 700])

    Online Experiment

    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # define the interval to epoch around a marker
    >>> ival = [0, 300]
    >>> while 1:
    ...     dat, mrk = amp.get_data()
    ...     newsamples = len(dat)
    ...     # the ringbuffer shall keep the last 2000 milliseconds,
    ...     # which is way bigger than our ival...
    ...     ringbuffer.append(dat, mrk)
    ...     cnt, mrk = ringbuffer.get()
    ...     # cnt contains now data up to 2000 millisecons, to make sure
    ...     # we don't see old markers again and again until they where
    ...     # pushed out of the ringbuffer, we need to tell segment_dat
    ...     # how many samples of cnt are actually new
    ...     epo = segment_dat(cnt, md, ival, newsamples=newsamples)

    """
    assert hasattr(dat, 'fs')
    assert hasattr(dat, 'markers')
    assert ival[0] <= ival[1]
    if newsamples is not None:
        assert newsamples >= 0
        # the times of the `newsamples`
        new_sample_times = dat.axes[timeaxis][-newsamples:] if newsamples > 0 else []
    # the expected length of each cnt in the resulted epo
    expected_samples = dat.fs * (ival[1] - ival[0]) / 1000
    data = []
    classes = []
    class_names = sorted(marker_def.keys())
    masks = []
    for t, m in dat.markers:
        for class_idx, classname in enumerate(class_names):
            if m in marker_def[classname]:
                mask = (t+ival[0] <= dat.axes[timeaxis]) & (dat.axes[timeaxis] < t+ival[1])
                # this one is quite expensive, we should move this stuff
                # out of the loop, or eliminate the loop completely
                # e.g. np.digitize for the marker to timeaxis mapping
                mask = np.flatnonzero(mask)
                if len(mask) != expected_samples:
                    # result is too short or too long, ignore it
                    continue
                # check if the new cnt shares at least one timepoint
                # with the new samples. attention: we don't only have to
                # check the ival but also the marker if it is on the
                # right side of the ival!
                times = dat.axes[timeaxis].take(mask)
                if newsamples is not None:
                    if newsamples == 0:
                        continue
                    if (len(np.intersect1d(times, new_sample_times)) == 0 and
                        t < new_sample_times[0]):
                        continue
                masks.append(mask)
                classes.append(class_idx)
    if len(masks) == 0:
        data = np.array([])
    else:
        # np.take inserts a new dimension at `axis`...
        data = dat.data.take(masks, axis=timeaxis)
        # we want that new dimension at axis 0 so we have to swap it.
        # before that we have to convert the netagive axis indices to
        # their equivalent positive one, otherwise swapaxis will be one
        # off.
        if timeaxis < 0:
            timeaxis = dat.data.ndim + timeaxis
        data = data.swapaxes(0, timeaxis)
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


def append(dat, dat2, axis=0, extra=None):
    """Append ``dat2`` to ``dat``.

    This method creates a copy of ``dat`` (with all attributes),
    concatenates ``dat.data`` and ``dat2.data`` along ``axis`` as well
    as ``dat.axes[axis]`` and ``dat2.axes[axis]``. If present, it will
    concatenate the attributes in ``extra`` as well and return the
    result.

    It also performs checks if the dimensions and lengths of ``data``
    and ``axes`` match and test if ``units`` and ``names`` are equal.

    Since ``append`` cannot know how to deal with the various attributes
    ``dat`` and ``dat2`` might have, it only copies the attributes of
    ``dat`` and deals with the attributes it knows about, namely:
    ``data``, ``axes``, ``names``, and ``units``.

    .. warning::
        This method is really low level and stupid. It does not know
        about markers or timeaxes, etc. it just appends two data
        objects. If you want to append continuous or epoched data
        consider using :func:`append_cnt` and :func:`append_epo`.

    Parameters
    ----------
    dat, dat2 : Data
    axis : int, optional
        the axis along which to concatenate. The default axis (0) does
        the right thing for continuous and epoched data as it
        concatenates along the time- or the class-axis respectively.
    extra : list of strings, optional
        a list of attributes in ``dat`` and ``dat2`` to concatenate as
        well. Currently the attributes must have the types ``list`` or
        ``ndarray``.

    Returns
    -------
    dat : Data
        a copy of ``dat`` with ``dat2`` appended

    Raises
    ------
    AssertionError
        if one of the following is true:
            * the dimensions of ``.data`` do not match
            * ``names`` are not equal
            * ``units`` are not equal
            * ``data.shape[i]`` are not equal for all i except ``i == axis``
            * ``axes[i]`` are not equal for all i except ``i == axis``
    TypeError
        * if one of the attributes in ``extra`` does not have the same
          type in ``dat`` and ``dat2``
        * if one of the attributes in ``extra`` has an unsupported type

    Examples
    --------

    >>> # concatenate two continuous data objects, and their markers,
    >>> # please note how the resulting marker is not correct, just
    >>> # appended
    >>> cnt.markers
    [[0, 'a'], [10, 'b']]
    >>> cnt2.markers
    [[20, 'c'], [30, 'd']]
    >>> cnt = append(cnt, cnt2, extra=['markers'])
    >>> cnt.markers
    [[0, 'a'], [10, 'b'], [20, 'c'], [30, 'd']]

    See Also
    --------
    append_cnt, append_epo

    """
    assert dat.data.ndim == dat2.data.ndim
    # convert negative axis to the equivalent positive one
    if axis < 0:
        axis = dat.data.ndim + axis
    for i in range(dat.data.ndim):
        assert dat.names[i] == dat2.names[i]
        assert dat.units[i] == dat2.units[i]
        if i == axis:
            continue
        assert dat.data.shape[i] == dat2.data.shape[i]
        assert np.all(dat.axes[i] == dat2.axes[i])
    data = np.concatenate([dat.data, dat2.data], axis=axis)
    axes = dat.axes[:]
    axes[axis] = np.concatenate([dat.axes[axis], dat2.axes[axis]])
    dat_new = dat.copy(data=data, axes=axes)
    if extra:
        for attr in extra:
            a1 = getattr(dat, attr)[:]
            a2 = getattr(dat2, attr)[:]
            if type(a1) != type(a2):
                raise TypeError('%s must have the same type' % attr)
            t = type(a1)
            if t == list:
                a1.extend(a2)
            elif t == np.ndarray:
                a1 = np.concatenate([a1, a2])
            else:
                raise TypeError('Concatenation of type %s is not supported.' % t)
            setattr(dat_new, attr, a1)
    return dat_new


def append_cnt(dat, dat2, timeaxis=-2, extra=None):
    """Append two continuous data objects.

    This method uses :func:`append` to append to continuous data
    objects. It also takes care that the resulting continuous will have
    a correct ``.axes[timeaxis]``. For that it uses the ``.fs``
    attribute and the length of the data to recalculate the timeaxis.

    If both ``dat`` and ``dat2`` have the ``markers`` attribute, the
    markers will be treated properly (i.e. by moving the markers of
    ``dat2`` by ``dat`` milliseconds to the right.

    Parameters
    ----------
    dat, dat2 : Data
    timeaxis : int, optional
    extra: list of strings, optional

    Returns
    -------
    dat : Data
        the resulting combination of ``dat`` and ``dat2``

    Raises
    ------
    AssertionError
        if at least one of the ``Data`` parameters has not the ``.fs``
        attribute or if the ``.fs`` attributes are not equal.

    See Also
    --------
    append, append_epo

    Examples
    --------

    >>> cnt.axis[0]
    [0, 1, 2]
    >>> cnt2.axis[0]
    [0, 1, 2]
    >>> cnt.fs
    1000
    >>> cnt = append_cnt(cnt, cnt2)
    >>> cnt.axis[0]
    [0, 1, 2, 3, 4, 5]

    """
    assert hasattr(dat, 'fs') and hasattr(dat2, 'fs')
    assert dat.fs == dat2.fs
    cnt = append(dat, dat2, axis=timeaxis, extra=extra)
    if hasattr(dat, 'markers') and hasattr(dat2, 'markers'):
        # move the markers from dat2 to the right by dat-milliseconds
        ms = dat.data.shape[timeaxis] / dat.fs * 1000
        markers1 = dat.markers[:]
        markers2 = [[x[0]+ms, x[1]] for x in dat2.markers]
        markers1.extend(markers2)
        cnt.markers = markers1
    # fix the timeaxis from 0, 1, 2, 0, 1, 2 -> 0, 1, 2, 3, 4, 5
    ms = cnt.data.shape[timeaxis] / cnt.fs * 1000
    cnt.axes[timeaxis] = np.linspace(0, ms, cnt.data.shape[timeaxis], endpoint=False)
    return cnt


def append_epo(dat, dat2, classaxis=0, extra=None):
    """Append two epoched data objects.

    This method just calls :func:`append`. In addition to the errors
    :func:`append` might throw, it will raise an error if the
    ``class_names`` are not equal if present in both objects.

    Parameters
    ----------
    dat, dat2 : Data
    classaxis : int, optional
    extra : list of strings, optional

    Returns
    -------
    dat : Data

    Raises
    ------
    ValueError
        if both objects have a ``class_names`` attribute, they must be
        equal

    See Also
    --------
    append, append_cnt

    Examples
    --------

    >>> epo = append_epo(epo, epo2)

    """
    if hasattr(dat, 'class_names') and hasattr(dat2, 'class_names'):
        if dat.class_names != dat2.class_names:
            raise ValueError('Incompatible class names.')
    epo = append(dat, dat2, axis=classaxis, extra=extra)
    return epo


def lfilter(dat, b, a, zi=None, timeaxis=-2):
    """Filter data using the filter defined by the filter coefficients.

    This method mainly delegates the call to
    :func:`scipy.signal.lfilter`.

    Parameters
    ----------
    dat : Data
        the data to be filtered
    b : 1-d array
        the numerator coefficient vector
    a : 1-d array
        the denominator coefficient vector
    zi : nd array, optional
        the initial conditions for the filter delay. If zi is ``None``
        or not given, initial rest is assumed.
    timeaxis : int, optional
        the axes in ``data`` to filter along to

    Returns
    -------
    dat : Data
        the filtered output

    See Also
    --------
    :func:`lfilter_zi`, :func:`filtfilt`, :func:`scipy.signal.lfilter`,
    :func:`scipy.signal.butter`, :func:`scipy.signal.butterord`

    Examples
    --------

    Generate and use a Butterworth bandpass filter for complete
    (off-line data):

    >>> # the sampling frequency of our data in Hz
    >>> dat.fs
    100
    >>> # calculate the nyquist frequency
    >>> fn = dat.fs / 2
    >>> # the desired low and high frequencies in Hz
    >>> f_low, f_high = 2, 13
    >>> # the order of the filter
    >>> butter_ord = 4
    >>> # calculate the filter coefficients
    >>> b, a = signal.butter(butter_ord, [f_low / fn, f_high / fn], btype='band')
    >>> filtered = lfilter(dat, b, a)

    Similar to the above this time in an on-line setting:

    >>> # pre-calculate the filter coefficients and the initial filter
    >>> # state
    >>> b, a = signal.butter(butter_ord, [f_low / fn, f_high / fn], btype='band')
    >>> zi = proc.lfilter_zi(b, a, len(CHANNELS))
    >>> while 1:
    ...     data, markers = amp.get_data()
    ...     # convert incoming data into ``Data`` object
    ...     cnt = Data(data, ...)
    ...     # filter the data, note how filter now also returns the
    ...     # filter state which we feed back into the next call of
    ...     # ``filter``
    ...     cnt, zi = lfilter(cnt, b, a, zi=zi)
    ...     ...

    """
    if zi is None:
        data = signal.lfilter(b, a, dat.data, axis=timeaxis)
        return dat.copy(data=data)
    else:
        data, zo = signal.lfilter(b, a, dat.data, zi=zi, axis=timeaxis)
        return dat.copy(data=data), zo


def filtfilt(dat, b, a, timeaxis=-2):
    """A forward-backward filter.

    Filter data twice, once forward and once backwards, using the filter
    defined by the filter coefficients.

    This method mainly delegates the call to
    :func:`scipy.signal.filtfilt`.

    Parameters
    ----------
    dat : Data
        the data to be filtered
    b : 1-d array
        the numerator coefficient vector
    a : 1-d array
        the denominator coefficient vector
    timeaxis : int, optional
        the axes in ``data`` to filter along to

    Returns
    -------
    dat : Data
        the filtered output

    See Also
    --------
    :func:`lfilter`

    Examples
    --------

    Generate and use a Butterworth bandpass filter for complete
    (off-line data):

    >>> # the sampling frequency of our data in Hz
    >>> dat.fs
    100
    >>> # calculate the nyquist frequency
    >>> fn = dat.fs / 2
    >>> # the desired low and high frequencies in Hz
    >>> f_low, f_high = 2, 13
    >>> # the order of the filter
    >>> butter_ord = 4
    >>> # calculate the filter coefficients
    >>> b, a = signal.butter(butter_ord, [f_low / fn, f_high / fn], btype='band')
    >>> filtered = filtfilt(dat, b, a)

    """
    # TODO: should we use padlen and padtype?
    data = signal.filtfilt(b, a, dat.data, axis=timeaxis)
    return dat.copy(data=data)


def lfilter_zi(b, a, n=1):
    """Compute an initial state ``zi`` for the :func:`lfilter` function.

    When ``n == 1`` (default), this method mainly delegates the call to
    :func:`scipy.signal.lfilter_zi` and returns the result ``zi``. If
    ``n > 1``, ``zi``  is repeated ``n`` times. This is useful if you
    want to filter n-dimensional data like multi channel EEG.

    Parameters
    ----------
    b, a : 1-d array
        The IIR filter coefficients
    n : int, optional
        The desired width of the output vector. If ``n == 1`` the output
        is simply the 1d zi vector. For ``n > 1``, the zi vector is
        repeated ``n`` times.

    Returns
    -------
    zi : n-d array
        The initial state of the filter.

    See Also
    --------
    :func:`lfilter`, :func:`scipy.signal.lfilter_zi`

    Examples
    --------

    >>> # pre-calculate the filter coefficients and the initial filter
    >>> # state
    >>> b, a = signal.butter(butter_ord, [f_low / fn, f_high / fn], btype='band')
    >>> zi = proc.lfilter_zi(b, a, len(CHANNELS))
    >>> while 1:
    ...     data, markers = amp.get_data()
    ...     # convert incoming data into ``Data`` object
    ...     cnt = Data(data, ...)
    ...     # filter the data, note how filter now also returns the
    ...     # filter state which we feed back into the next call of
    ...     # ``filter``
    ...     cnt, zi = lfilter(cnt, b, a, zi=zi)
    ...     ...

    """
    zi = signal.lfilter_zi(b, a)
    if n > 1:
        zi = np.tile(zi, (n, 1)).T
    return zi


def clear_markers(dat, timeaxis=-2):
    """Remove markers that are outside of the ``dat`` time interval.

    This method removes the markers that are out of the time interval
    described in the ``dat`` object.

    If the ``dat`` object has not ``markers`` attribute or the markers
    are empty, simply a copy of ``dat`` is returned.

    If ``dat.data`` is empty, but has markers, all markers are removed.

    Parameters
    ----------
    dat : Data
    timeaxis : int, optional

    Returns
    -------
    dat : Data
        a copy of the Data object, with the respective markers removed

    Raises
    ------
    AssertionError
        if the given ``dat`` has not ``fs`` attribute

    Examples
    --------

    >>> dat.axes[0]
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> dat.fs
    1000
    >>> dat.markers
    [[-6, 'a'], [-5, 'b'], [0, 'c'], [4.9999, 'd'], [5, 'e']]
    >>> dat = clear_markers(dat)
    >>> dat.markers
    [[-5, 'b'], [0, 'c'], [4.9999, 'd']]

    """
    if not hasattr(dat, 'markers') or not dat.markers:
        # nothing to do, we don't have any markers
        return dat.copy()
    if not dat:
        # we don't have any data, and thus no time interval, we remove
        # all markers
        logger.warning('Removing Marker from empty Data, this might be an error.')
        return dat.copy(markers=[])
    assert hasattr(dat, 'fs')
    sample_len = 1000 / dat.fs
    markers = dat.markers[:]
    min, max = dat.axes[timeaxis][0], dat.axes[timeaxis][-1] + sample_len
    markers = [x for x in markers if min <= x[0] < max]
    return dat.copy(markers=markers)


def select_ival(dat, ival, timeaxis=-2):
    """Select interval from data.

    This method selects the time segment(s) defined by ``ival``. It will
    also automatically remove markers outside of the desired interval in
    the returned Data object.

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
    dat = dat.copy(data=data, axes=axes)
    if hasattr(dat, 'markers'):
        dat.markers = [x for x in dat.markers if ival[0] <= x[0] < ival[1]]
    return dat


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
    AssertionError
        if ``dat`` has no ``.class_names`` attribute.

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
    AssertionError
        if ``dat`` has no ``.class_names`` attribute.

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

    .. note::
        Note that this method does not low-pass filter the data before
        sub-sampling.

    .. note::
        If you use this method in an on-line setting (i.e. where you
        process the data in chunks and not as a whole), you should make
        sure that ``subsample`` does not drop "half samples" by ensuring
        the source data's length is in multiples of the target data's
        sample length.

        Let's assume your source data is sampled in 1kHz and you want to
        subsample down to 100Hz. One sample of the source data is 1ms
        long, while the target samples will be 10ms long. In order to
        ensure that ``subsample`` does not eat fractions of samples at
        the end of your data, you have to make sure that your source
        data is multiples of 10ms (i.e. 1010, 1020, etc) long. You might
        want to use :class:`wyrm.types.BlockBuffer` for this (see
        Examples below).


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
    lfilter

    Examples
    --------

    Load some EEG data with 1kHz, bandpass filter it and downsample it
    to 100Hz.

    >>> dat = load_brain_vision_data('some/path')
    >>> dat.fs
    1000.0
    >>> fn = dat.fs / 2 # nyquist frequ
    >>> b, a = butter(4, [8 / fn, 40 / fn], btype='band')
    >>> dat = lfilter(dat, b, a)
    >>> dat = subsample(dat, 100)
    >>> dat.fs
    100.0

    Online Experiment

    >>> bbuffer = BlockBuffer(10) # 10 ms is the target block size
    >>> while 1:
    ...     cnt = ... # get 1kHz continous data from your amp
    ...     # put the data into the block buffer
    ...     # bbget will onlry return the data in multiples of 10ms or
    ...     # nothing
    ...     bbuffer.append(cnt)
    ...     cnt = bbuffer.get()
    ...     if not cnt:
    ...         continue
    ...     # filter, etc
    ...     subsample(cnt, 100)

    Raises
    ------
    AssertionError
        * if ``freq`` is not a whole number divisor of ``dat.fs``
        * if ``dat`` has no ``.fs`` attribute
        * if ``dat.data.shape[timeaxis] != len(dat.axes[timeaxis])``

    """
    assert hasattr(dat, 'fs')
    assert dat.data.shape[timeaxis] == len(dat.axes[timeaxis])
    assert dat.fs % freq == 0
    factor = int(dat.fs / freq)
    if dat.data.shape[timeaxis] % factor != 0:
        logger.warning('Subsampling led to loss of %i samples, in an online setting consider using a BlockBuffer with a buffer size of a multiple of %i samples.' % (dat.data.shape[timeaxis] % factor, factor))
    idxmask = np.arange(dat.data.shape[timeaxis], step=factor)
    data = dat.data.take(idxmask, timeaxis)
    axes = dat.axes[:]
    axes[timeaxis] =  axes[timeaxis].take(idxmask)
    return dat.copy(data=data, axes=axes, fs=freq)


def spectrum(dat, timeaxis=-2):
    """Calculate the spectrum of a data object.

    This method performs a fast fourier transform on the data along the
    timeaxis and returns a new `Data` object which is transformed into
    the frequency domain. The values are the amplitudes of of the
    respective frequencies.

    Parameters
    ----------
    dat : Data
        Data object with `.fs` attribute
    timeaxis : int, optional
        axis to perform the fft along

    Returns
    -------
    dat : Data
        Data object with the timeaxis transformed into the frequency
        domain. The values of the spectrum are the amplitudes of the
        respective frequencies.

    Examples
    --------
    >>> # dat can be continuous or epoched
    >>> dat.axes
    ['time', 'channel']
    >>> spm = spectrum(dat)
    >>> spm.axes
    ['frequency', 'channel']

    Raises
    ------
    AssertionError
        if the ``dat`` parameter has no ``.fs`` attribute

    See Also
    --------
    spectrogram, stft

    """
    # oh look at that! a dumb idea just found a friend.
    assert hasattr(dat, 'fs')
    # number of samples of our data
    length = dat.data.shape[timeaxis]
    fourier = sp.fftpack.fft(dat.data, axis=timeaxis)
    fourier = fourier.take(np.arange(length)[1:int(length/2)], axis=timeaxis)
    amps = 2 * fourier / length
    amps = np.abs(amps)
    freqs = sp.fftpack.fftfreq(length, 1/dat.fs)
    freqs = freqs[1:int(length/2)]
    axes = dat.axes[:]
    axes[timeaxis] = freqs
    names = dat.names[:]
    names[timeaxis] = 'frequency'
    units = dat.units[:]
    units[timeaxis] = 'dl'
    # TODO: units in original unit or dimensionles?)
    spm = dat.copy(data=amps, axes=axes, names=names, units=units)
    delattr(spm, 'fs')
    return spm


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


def calculate_csp(epo, classes=None):
    """Calculate the Common Spatial Pattern (CSP) for two classes.

    This method calculates the CSP and the corresponding filters. Use
    the columns of the patterns and filters.

    Examples
    --------
    Calculate the CSP for the first two classes::

    >>> w, a, d = calculate_csp(epo)
    >>> # Apply the first two and the last two columns of the sorted
    >>> # filter to the data
    >>> filtered = apply_csp(epo, w, [0, 1, -2, -1])
    >>> # You'll probably want to get the log-variance along the time
    >>> # axis, this should result in four numbers (one for each
    >>> # channel)
    >>> filtered = np.log(np.var(filtered, 0))

    Select two classes manually::

    >>> w, a, d = calculate_csp(epo, [2, 5])

    Parameters
    ----------
    epo : epoched Data object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    classes : list of two ints, optional
        If ``None`` the first two different class indices found in
        ``epo.axes[0]`` are chosen automatically otherwise the class
        indices can be manually chosen by setting ``classes``

    Returns
    -------
    v : 2d array
        the sorted spacial filters
    a : 2d array
        the sorted spacial patterns. Column i of a represents the
        pattern of the filter in column i of v.
    d : 1d array
        the variances of the components

    Raises
    ------
    AssertionError :
        If:
          * ``classes`` is not ``None`` and has less than two elements
          * ``classes`` is not ``None`` and the first two elements are
            not found in the ``epo``
          * ``classes`` is ``None`` but there are less than two
            different classes in the ``epo``

    See Also
    --------
    :func:`apply_csp`, :func:`calculate_spoc`

    References
    ----------
    http://en.wikipedia.org/wiki/Common_spatial_pattern

    """
    n_channels = epo.data.shape[-1]
    if classes is None:
        # automagically find the first two different classidx
        # we don't use uniq, since it sorts the classidx first
        # first check if we have a least two diffeent idxs:
        assert len(np.unique(epo.axes[0])) >= 2
        cidx1 = epo.axes[0][0]
        cidx2 = epo.axes[0][epo.axes[0] != cidx1][0]
    else:
        assert (len(classes) >= 2 and
            classes[0] in epo.axes[0] and
            classes[1] in epo.axes[0])
        cidx1 = classes[0]
        cidx2 = classes[1]
    epoc1 = select_epochs(epo, np.nonzero(epo.axes[0] == cidx1)[0], classaxis=0)
    epoc2 = select_epochs(epo, np.nonzero(epo.axes[0] == cidx2)[0], classaxis=0)
    # we need a matrix of the form (observations, channels) so we stack trials
    # and time per channel together
    x1 = epoc1.data.reshape(-1, n_channels)
    x2 = epoc2.data.reshape(-1, n_channels)
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


def apply_csp(epo, filt, columns=[0, -1]):
    """Apply the CSP filter.

    Apply the spacial CSP filter to the epoched data.

    Parameters
    ----------
    epo : epoched ``Data`` object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    filt : 2d array
        the CSP filter (i.e. the ``v`` return value from
        :func:`calculate_csp`)
    columns : array of ints, optional
        the columns of the filter to use. The default is the first and
        the last one.

    Returns
    -------
    epo : epoched ``Data`` object
        The channels from the original have been replaced with the new
        virtual CSP channels.

    Examples
    --------

    >>> w, a, d = calculate_csp(epo)
    >>> epo = apply_csp(epo, w)

    See Also
    --------
    :func:`calculate_csp`

    """
    f = filt[:, columns]
    data = np.array([np.dot(epo.data[i], f) for i in range(epo.data.shape[0])])
    axes = epo.axes[:]
    axes[-1] = np.array(['csp %i' % i for i in range(data.shape[-1])])
    names = epo.names[:]
    names[-1] = 'CSP Channel'
    dat = epo.copy(data=data, axes=axes, names=names)
    return dat


def calculate_spoc(epo):
    """Compute source power co-modulation analysis (SPoC)

    Computes spatial filters that optimize the co-modulation (here
    covariance) between the epoch-wise variance (as a proxy for spectral
    power) and a given target signal.

    This SPoc function returns a full set of components (i.e. filters
    and patterns) of which the first component maximizes the
    co-modulation (i.e. positive covariance) and the last component
    minimizes it (i.e. maximizes negative covariance).

    .. note:: Since the covariance is optimized, it may be affected by
        outliers in the data (i.e. trials/epochs with very large
        variance that is due to artifacts). Please remove be sure to
        remove these epochs if possible before calling this function!

    Parameters
    ----------
    epo : epoched Data oject
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel. The data in epo
        should be band-pass filtered for the frequency band of interest.
        The values of the target variable (i.e. ``epo.axes[0]``) must be
        present in epo.

    Returns
    -------
    v : 2d array
        The spatial filters optimized by the ``SPoC_lambda`` algorithm.
        Each column in the matrix is a filter.
    a : 2d array
        The spatial activation patterns that correspond to the filters
        in ``v``. Each column is a spatial pattern. when visualizing the
        SPoC components as scalp maps, plot the spatial patterns and not
        the filters. See also [2]_.
    d : 1d array
        The lambda values that correspond to the filters/patterns in
        ``v`` and ``a``, sorted from largest (positive covariance) to
        smallest (negative covariance).

    Examples
    --------

    Split data in training and test set

    Calculate SPoC::

    >>> w, a, d = calculate_spoc(epo)

    Identify the components with strongest co-modulation by checking the
    covariance values stored in ``d``. If there is positive covariance
    with the target variable it will be the first, otherwise the last:

    >>> w = w[:, 0]

    Apply the filter(s) to the test data::

    >>> filtered = np.dot(data, w)

    Notes
    -----

    SPoC assumes that there is a linear relationship between a measured
    target signal and the dynamics of the spectral power of an
    oscillatory source that is hidden in the data. The target signal my
    be a stimulus property (e.g. intensity, frequency, color, ...), a
    behavioral measure (e.g. reaction times, ratings, ...) , or any
    other uni-variate signal of interest. The time-course of spectral
    power of the oscillatory source signal is approximated by variance
    across small time segments (epochs). Thus, if the power of a
    specific frequency band is investigated, the input signals must be
    band-passed filtered before they are segmented into epochs and given
    to this function. This method implements ``SPoC_lambda``, presented
    in [1]_. Thus, source activity is extracted from the input data via
    spatial filtering. The spatial filters are optimized such that the
    epoch-wise variance maximally covaries with the given target signal
    ``z``.

    See Also
    --------
    :func:`calculate_csp`


    References
    ----------

    .. [1] S. Dähne, F. C. Meinecke, S. Haufe, J. Höhne, M. Tangermann,
        K. R. Müller, V. V. Nikulin "SPoC: a novel framework for
        relating the amplitude of neuronal oscillations to behaviorally
        relevant parameters", NeuroImage, 86(0):111-122, 2014

    .. [2] S. Haufe, F. Meinecke, K. Görgen, S. Dähne, J. Haynes, B.
        Blankertz, F. Biessmann, "On the interpretation of weight
        vectors of linear models in multivariate neuroimaging",
        NeuroImage, 87:96-110, 2014

    """
    z = epo.axes[0][:]
    # convert into float just in case, because in the BCI case we'll
    # deal mostly with integers (i.e. the class indices) here
    z = z.astype(np.float)

    z -= z.mean()
    z /= z.std()

    ce = np.empty([epo.data.shape[0], epo.data.shape[2], epo.data.shape[2]])
    for i in range(epo.data.shape[0]):
        ce[i] = np.cov(epo.data[i].transpose())

    c = ce.mean(axis=0)
    for i in range(ce.shape[0]):
        ce[i] = ce[i] * z[i]
    cz = ce.mean(axis=0)

    # solution of csp objective via generalized eigenvalue problem
    # in matlab the signature is v, d = eig(a, b)
    d, v = sp.linalg.eig(cz, c)
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
    AssertionError
        if the ``dat`` has no ``.class_names`` attribute.

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


def calculate_cca(dat_x, dat_y, timeaxis=-2):
    """Calculate the Canonical Correlation Analysis (CCA).

    This method calculates the canonical correlation coefficient and
    corresponding weights which maximize a correlation coefficient
    between linear combinations of the two specified multivariable
    signals.

    Parameters
    ----------
    dat_x, dat_y : continuous Data object
        these data should have the same length on the time axis.
    timeaxis : int, optional
        the index of the time axis in ``dat_x`` and ``dat_y``.

    Returns
    -------
    rho : float
        the canonical correlation coefficient.
    w_x, w_y : 1d array
        the weights for mapping from the specified multivariable signals
        to canonical variables.

    Raises
    ------
    AssertionError :
        If:
          * ``dat_x`` and ``dat_y`` is not continuous Data object
          * the length of ``dat_x`` and ``dat_y`` is different on the
            ``timeaxis``

    Examples
    --------
    Calculate the CCA of the specified multivariable signals.

    >>> rho, w_x, w_y = calculate_cca(dat_x, dat_y)
    >>> # Calculate canonical variables via obtained weights
    >>> cv_x = np.dot(dat_x, w_x)
    >>> cv_y = np.dot(dat_y, w_y)

    References
    ----------
    http://en.wikipedia.org/wiki/Canonical_correlation

    """
    assert (len(dat_x.data.shape) == len(dat_y.data.shape) == 2 and
        dat_x.data.shape[timeaxis] == dat_y.data.shape[timeaxis])

    if timeaxis == 0 or timeaxis == -2:
        x = dat_x.data.copy()
        y = dat_y.data.copy()
    else:
        x = dat_x.data.T.copy()
        y = dat_y.data.T.copy()

    # calculate covariances and it's inverses
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    N = x.shape[0]
    c_xx = np.dot(x.T, x) / N
    c_yy = np.dot(y.T, y) / N
    c_xy = np.dot(x.T, y) / N
    c_yx = np.dot(y.T, x) / N
    ic_xx = np.linalg.pinv(c_xx)
    ic_yy = np.linalg.pinv(c_yy)
    # calculate w_x
    w, v = np.linalg.eig(functools.reduce(np.dot, [ic_xx, c_xy, ic_yy, c_yx]))
    w_x = v[:, np.argmax(w)].real
    w_x = w_x / np.sqrt(functools.reduce(np.dot, [w_x.T, c_xx, w_x]))
    # calculate w_y
    w, v = np.linalg.eig(functools.reduce(np.dot, [ic_yy, c_yx, ic_xx, c_xy]))
    w_y = v[:, np.argmax(w)].real
    w_y = w_y / np.sqrt(functools.reduce(np.dot, [w_y.T, c_yy, w_y]))
    # calculate rho
    rho = abs(functools.reduce(np.dot, [w_x.T, c_xy, w_y]))
    return rho, w_x, w_y


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
    # and calculate the mean along the sampling axis
    means = np.mean(dat.data.compress(mask, timeaxis), axis=timeaxis)
    data = dat.data - np.expand_dims(means, timeaxis)
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
    axes = dat.axes[:2]
    axes[-1] = np.arange(data.shape[-1])
    names = dat.names[:2]
    names[-1] = 'feature vector'
    units = dat.units[:2]
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
    return np.sign(r) * np.square(r)


def logarithm(dat):
    """Computes the element wise natural logarithm of ``dat.data``.

    Calling this method is equivalent to calling

    >>> dat.copy(data=np.log(dat.data))

    Parameters
    ----------
    dat : Data
        a Data object

    Returns
    -------
    dat : Data
        a copy of ``dat`` with the element wise natural logarithms of
        the values in ``.data``

    See Also
    --------
    :func:`square`

    """
    data = np.log(dat.data)
    return dat.copy(data=data)


def square(dat):
    """Computes the element wise square of ``dat.data``.

    Calling this method is equivalent to calling

    >>> dat.copy(data=np.square(dat.data))

    Parameters
    ----------
    dat : Data
        a Data object


    Returns
    -------
    dat : Data
        a copy of ``dat`` with the element wise squares of the values in
        ``.data``

    See Also
    --------
    :func:`logarithm`

    """
    data = np.square(dat.data)
    return dat.copy(data=data)


def variance(dat, timeaxis=-2):
    """Compute the variance along the ``timeaxis`` of ``dat``.

    This method reduces the dimensions of `dat.data` by one.

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
    axes = dat.axes[:]
    axes.pop(timeaxis)
    names = dat.names[:]
    names.pop(timeaxis)
    units = dat.units[:]
    units.pop(timeaxis)
    return dat.copy(data=data, axes=axes, names=names, units=units)

