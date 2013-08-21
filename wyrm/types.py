
"""Data type definitions.

This module provides the basic data types for Wyrm.

"""


from __future__ import division

import copy
import logging

import numpy as np


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class Data(object):
    """Generic, self-describing data container.

    This data structure is very generic on purpose. The goal here was to
    provide something which can fit the various different known and yet
    unknown requirements for BCI algorithms.

    At the core of ``Data`` is its n-dimensional ``.data`` attribute
    which holds the actual data. Along with the data, there is meta
    information about each axis of the data, contained in ``.axes``,
    ``.names``, and ``.units``.

    Most toolbox methods rely on a *convention* how specific data should
    be structured (i.e. they assume that the channels are always in the
    last dimension). You don't have to follow this convention (or
    sometimes it might not even be possible when trying out new things),
    and all methods, provide an optional parameter to tell them on which
    axis they should work on.

    Continuous Data:
        Continuous Data is usually EEG data and consists of a 2d array
        ``[time, channel]``. Whenever you have continuous data, time and
        channel should be the last two dimensions.

    Epoched Data:
        Epoched data can be seen as an array of (non-epoched) data. The
        epoch should always be the first dimension. Most commonly used is
        epoched continuous EEG data which looks like this: ``[class,
        time, channel]``.

    Feature Vector:
        Similar to Epoched Data, with classes in the first dimension.

    A :func:`__eq__` function is providet to test for equality of two
    Data objects (via ``==``). This method only checks for the known
    attributes and does not guaranty correct result if the Data object
    contains custom attributes. It is mainly used in unittests.

    Parameters
    ----------
    data : ndarray
    axes : nlist of 1darrays
    names : nlist of strings
    units : nlist of strings

    Attributes
    ----------
    data : ndarray
        n-dimensional data array
    axes : nlist of 1darrays
        each element of corresponds to a dimension of ``.data`` (i.e.
        the first one in ``.axes`` to the first dimension in ``.data``
        and so on). The 1-dimensional arrays contain the description of
        the data along the appropriate axis in ``.data``. For example if
        ``.data`` contains Continuous Data, then ``.axes[0]`` should be
        an array of timesteps and ``.axes[1]`` an array of channel names
    names : nlist of strings
        the human readable description of each axis, like 'time', or 'channel'
    units : nlist of strings
        the human readable description of the unit used for the data in
        ``.axes``

    """
    def __init__(self, data, axes, names, units):
        """Initialize a new ``Data`` object.

        Upon initialization we check if ``axes``, ``names``, and
        ``units`` have the same length and if their respective length
        matches the shape of ``data``.

        Raises
        ------
        AssertionError : if the lengths of the parameters are not
            correct.

        """
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
        assumes equality if those are equal. This method is heavily used
        in unittests.

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


