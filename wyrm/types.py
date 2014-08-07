
"""Data type definitions.

This module provides the basic data types for Wyrm, like the
:class:`Data` and :class:`RingBuffer` classes.

"""


from __future__ import division

import copy
import logging

import numpy as np

from wyrm.processing import append_cnt


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

    :meth:`Data.__eq__` and :meth:`Data.__ne__` functions are provided
    to test for equality of two Data objects (via ``==`` and ``!=``).
    This method only checks for the known attributes and does not
    guaranty correct result if the Data object contains custom
    attributes. It is mainly used in unittests.

    Parameters
    ----------
    data : ndarray
    axes : nlist of 1darrays
    names : nlist of strings
    units : nlist of strings

    Attributes
    ----------
    data : ndarray
        n-dimensional data array if the array is empty
        (i.e. ``data.size == 0``), the ``Data`` object is assumed to be
        empty
    axes : nlist of 1-darrays
        each element of corresponds to a dimension of ``.data`` (i.e.
        the first one in ``.axes`` to the first dimension in ``.data``
        and so on). The 1-dimensional arrays contain the description of
        the data along the appropriate axis in ``.data``. For example if
        ``.data`` contains Continuous Data, then ``.axes[0]`` should be
        an array of timesteps and ``.axes[1]`` an array of channel names
    names : nlist of strings
        the human readable description of each axis, like 'time', or
        'channel'
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
        AssertionError
            if the lengths of the parameters are not correct.

        """
        if data.size == 0:
            pass
        else:
            assert data.ndim == len(axes) == len(names) == len(units)
            assert [len(a) for a in axes] == list(data.shape)
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
        # check if both have the same attributes
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        # .data
        if not np.array_equal(self.data, other.data):
            return False
        # .axes
        if len(self.axes) != len(other.axes):
            return False
        for i in range(len(self.axes)):
            if self.axes[i].shape != other.axes[i].shape:
                return False
            if not (self.axes[i] == other.axes[i]).all():
                return False
        # .names
        if self.names != other.names:
            return False
        # .units
        if self.units != other.units:
            return False
        # optional extra attributes
        if hasattr(self, 'markers') and self.markers != other.markers:
            return False
        if hasattr(self, 'fs') and self.fs != other.fs:
            return False
        # the stuff we care about seems to be equal, this does not mean
        # the rest we didn't check is, but anyways...
        return True

    def __ne__(self, other):
        """Test for inequality.

        If :func:`__eq__` is implemented and :func:`__ne__` is not,
        strange comparisons evaluate to True like:

        >>> d1 == d2 and d1 != d2

        This method just returns the negation of :meth:`__eq__`. So the
        same restrictions of :meth:`__eq__` about its reliability apply.

        Parameters
        ----------
        other : Data

        Returns
        -------
        equal : Boolean
            True if ``self`` and ``other`` are not equal, False
            otherwise.

        """
        return not self.__eq__(other)

    def __nonzero__(self):
        """Return the truth value for the object instance.

        Similar to Python's built in types we return ``False`` if the
        data instance is empty and ``True`` otherwise. Please note that
        we only check for the size of ``.data`` and ignore other
        attributes like ``.markers`` which might not be empty.

        Examples
        --------

        Easy checking if a data object contains data or not:

        >>> if not cnt:
        ...     continue

        is equivalent to:

        >>> if cnt.data.size == 0:
        ...     continue

        Returns
        -------
        nonzero : int
            ``self.data.size``

        """
        return self.data.size

    # This method was added for Python3 compatibility
    def __bool__(self):
        """Return truth value of the object instance.

        This method returns False if the __nonzero__ value is 0 else
        True.

        Returns
        -------
        truth : Bool
            ``False`` if :func:`__nonzero__` was ``0``, else ``True``.

        See Also
        --------
        :func:`__nonzero__`

        """
        return False if self.__nonzero__() == 0 else True

    def __str__(self):
        """Human readable representation for a data object.

        Returns
        -------
        str : str
            a human readable representation of the data object

        """
        data = 'Data: \n%s' % self.data
        axes = 'Axes: \n%s' % self.axes
        names = 'Names: \n%s' % self.names
        units = 'Units: \n%s' % self.units
        return '\n'.join([data, axes, names, units])

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
        for name, value in list(kwargs.items()):
            setattr(obj, name, value)
        return copy.deepcopy(obj)


class RingBuffer(object):
    """Circular Buffer implementation.

    This implementation has a guaranteed upper bound for read and write
    operations as well as a constant memory usage, which is the size of
    the maximum length of the buffer in memory.

    Reading and writing will take at most the time it takes to copy a
    continuous chunk of length ``MAXLEN`` in memory. E.g. for the
    extreme case of storing the last 60 seconds of 64bit data, sampled
    with 1kHz and 128 channels (~60MB), reading a full buffer will take
    ~25ms, as well as writing when storing more than than 60 seconds at
    once. Writing will be usually much faster, as one stores usually
    only a few milliseconds of data per run. In that case writing will
    be a fraction of a millisecond.

    Parameters
    ----------
    length_ms : int
        the length of the ring buffer in milliseconds

    Attributes
    ----------
    length_ms : int
        the length of the ring buffer in milliseconds
    length : int
        the length of the ring buffer in samples
    data : ndarray
        the contents of the ring buffer, you should not read or write
        this attribute directly but via the :meth:`RingBuffer.get` and
        :meth:`RingBuffer.append` methods
    markers : array of [int, str]
        the markers belonging to the data currently in the ring buffer
    full : boolean
        indicates if the buffer has at least ``length`` elements stored
    idx : int
        the starting position of the oldest data in the ring buffer

    Examples
    --------

    >>> rb = RingBuffer(length)
    >>> while True:
    ...     rb.append(amp.get_data())
    ...     buffered = rb.get()
    ...     # do something with buffered


    """

    def __init__(self, length_ms):
        """Initialize the Ringbuffer.

        Parameters
        ----------
        length : int
            the length of the ring buffer in milliseconds

        """
        # the maximum length of the ring buffer in ms
        self.length_ms = length_ms
        # the length of the buffer in samples
        self.length = None
        self.data = None
        self.markers = []
        self.axes = []
        self.units = []
        self.names = []
        self.fs = None
        # indicate if the buffer write was wrapped around at least once
        self.full = False
        # the index where to insert new data (= the start of the oldest
        # data)
        self.idx = 0

    def _move_markers(self, markers, steps):
        """Move marker `steps` samples to the left or right.

        This method respects the sampling frequency of the data.

        Parameters
        ----------
        markers : list of (float, str)
        steps : int
            the number of samples to move the markers (a negative value
            moves the indices to the left)

        Returns
        -------
        markers : list of (float, str)

        """
        shift_ms = 1000 / self.fs * steps
        return [[x[0] + shift_ms, x[1]] for x in markers]

    def append(self, dat):
        """Append data to the Ringbuffer, overwriting old data if necessary.

        Parameters
        ----------
        dat : Data
            a continuous data object

        Raises
        ------
        ValueError
            if the [1:]-dimensions (all but the first one) of ``data``
            does not match the ring buffer dimensions

        """
        assert hasattr(dat, 'markers')
        assert hasattr(dat, 'fs')
        data = dat.data.copy()
        markers = dat.markers[:]
        # we have nothing to append
        if len(data) == 0:
            if markers:
                logger.warning('Received Empty Data with markers. Discarding markers.')
                logger.warning(markers)
            return
        # we append the first time, initialize .data with the correct
        # shape
        if self.data is None:
            self.fs = dat.fs
            self.length = self.length_ms / 1000 * self.fs
            if not self.length.is_integer():
                logger.error('Length is not an integer, please check length_ms and fs. Rounding errors will lead to loss of samples.')
            self.length = int(self.length)
            buffershape = list(data.shape)
            buffershape[0] = self.length
            self.data = np.empty(buffershape)
            self.axes = dat.axes[:]
            self.axes[0] = np.linspace(0, 1000 * self.length / self.fs, self.length, endpoint=False)
            self.names = dat.names[:]
            self.units = dat.units[:]
        # incoming data is bigger than the buffer's capacity
        if len(data) > self.length:
            logger.warning('Discarding data that was longer than the ring buffer.')
            surplus = len(data) - self.length
            data = data[surplus:]
            markers = self._move_markers(markers, -surplus)
        # the markers, please be careful when changing it, this is quite
        # tricky:
        # size of the buffer (0..self.length-1)
        size = self.length if self.full else self.idx
        # append the new markers to the end of the existing ones,
        # shifting the new indices by 'size'
        markers = self._move_markers(markers, size)
        self.markers.extend(markers)
        # if we wrapped around, move all elements to the left by the
        # size of the surplus elements
        if size + len(data) > self.length:
            move = self.length - (size + len(data))
            self.markers = self._move_markers(self.markers, move)
        self.markers = [x for x in self.markers if x[0] >= 0]
        # /end of markers
        # we can write without wrapping around the buffer's end
        if self.idx + len(data) < self.length:
            self.data[self.idx:self.idx+len(data)] = data
            self.idx += len(data)
        # we will wrap around the buffer's end
        else:
            self.full = True
            l1 = self.length - self.idx
            l2 = len(data) - l1
            self.data[-l1:] = data[:l1]
            self.data[:l2] = data[l1:]
            self.idx = l2

    def get(self):
        """Get all buffered data.

        The returned data will have *at most* the length of ``length``.

        Returns
        -------
        data : Data
            the full contents of the ring buffer

        """
        # no data has ever been appended to this ring buffer
        if self.data is None:
            data = np.array([])
            axes = []
        # the ringbuffer wrapped around at least once
        elif self.full:
            data = np.concatenate([self.data[self.idx:], self.data[:self.idx]], axis=0)
            axes = self.axes[:]
        # the ring buffer hasn't been filled completely yet
        else:
            data = self.data[:self.idx].copy()
            axes = self.axes[:]
            axes[0] = axes[0][:self.idx]
        d = Data(data=data, axes=axes, names=self.names[:], units=self.units[:])
        d.markers = self.markers[:]
        d.fs = self.fs
        return d


class BlockBuffer(object):
    """A buffer that returns data chunks in multiples of a block length.

    This buffer is a first-in-first-out (FIFO) buffer that returns data
    in multiples of a desired block length. The block length is defined
    in samples.

    Parameters
    ----------
    samples : int, optional
        the desired block length in samples

    Examples
    --------

    >>> bbuffer = BlockBuffer(10)
    >>> ...
    >>> while 1:
    ... cnt = some_aquisition_method()
    ... # How to use the BlockBuffer
    ... bbuffer.append(cnt)
    ... cnt = bbuffer.get()
    ... if not cnt:
    ...     continue
    ... # after here cnt is guaranteed to be in multiples of 10 samples

    """

    def __init__(self, samples=50):
        """Initialize the Block Buffer.

        Parameters
        ----------
        samples : int, optional
            the desired block length in samples

        """
        self.samples = samples
        self.dat = None

    def append(self, dat):
        """Append data to the Block Buffer.

        This method accumulates the incoming data.

        Parameters
        ----------
        dat : Data
            continuous Data object

        """
        if self.dat is None:
            self.dat = dat.copy()
        elif not dat:
            pass
        else:
            self.dat = append_cnt(self.dat, dat)

    def get(self):
        """Pop the contents of the Block Buffer.

        The data returned has a length of multiples of ``samples``. If
        there is a fraction of ``samples`` data more in the buffer, that
        data is kept and future :meth:`append` operations will append
        new data to it.

        Returns
        -------
        dat : Data
            continuous Data object

        """
        if self.dat is None or self.dat.data.shape[0] < self.samples:
            return Data(np.array([]), [], [], [])
        if self.dat.data.shape[0] % self.samples == 0:
            ret = self.dat.copy()
            self.dat = None
            return ret
        else:
            marker_orig = self.dat.markers[:]
            # number of samples to return
            n = (self.dat.data.shape[0] // self.samples) * self.samples
            # first part
            dat1 = self.dat.copy()
            dat1.data = dat1.data[:n]
            dat1.axes[0] = dat1.axes[0][:n]
            # remaining (incomplete) part
            dat2 = self.dat.copy()
            dat2.data = dat2.data[n:]
            dat2.axes[0] = dat2.axes[0][n:]
            # split the markers
            t0 = dat2.axes[0][0]
            dat1.markers = [x for x in self.dat.markers if x[0] < t0]
            dat2.markers = [x for x in self.dat.markers if x[0] >= t0]
            # align the second part to t0
            dat2.axes[0] -= t0
            dat2.markers = [[x[0] - t0, x[1]] for x in dat2.markers]
            self.dat = dat2
            return dat1

