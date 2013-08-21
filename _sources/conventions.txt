Conventions Used in This Toolbox
================================

The idea is to make the regular use cases easy and the hard ones possible.

Common Vocabulary
-----------------

A common vocabulary is important when different parties talk about complicated
concepts to make sure everyone fully understands what the other is talking
about.

Dimensions and Axes
~~~~~~~~~~~~~~~~~~~

Talking about dimensions in context of numpy arrays can be a bit confusing
especially when coming from a mathematical background. We use the following
convention: A point in the 3D Space ``(x, y, z)`` is an array with one dimension
of length 3. An array of ``n`` such points would be an array with two
dimensions, the first axis (dimension) with the length of ``n``, and the exond
axis with the length of ``3``.

::

    >>> import numpy as np
    >>> a = np.arange(20)
    >>> a
    array([ 0,  1,  2,  3, ..., 17, 18, 19])
    >>> a.ndim
    1              # one dimension (or axis)
    >>> a.shape
    (20,)          # of lenght 20

    >>> a = a.reshape(4, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    >>> a.ndim
    2              # two dimensions
    >>> a.shape
    (4, 5)         # of length 4 and 5


Data Structures
---------------

Wyrm uses one data structure :class:`wyrm.types.Data` to wrap the different data
during the processing. `Data` is very generic and thus flexible on purpose. It
can contain all kinds of data and tries to stay out of the way. `Data` is also
self-explaining in a sense that it does not only contain the raw data but also
meta-data about the axes, like names and units and the values of the axes (for a
complete overview on `Data` please refer to the documentation).

Most of Wyrm's toolbox methods are expecting a ``Data`` object as an argument.
Since ``Data`` is very flexible and does not impose for example the order of the
axes, it is important to abide a certain convention:

.. glossary::

    Continuous Data
        Continuous Data is usually (but not limited to) raw EEG data. It has two
        axes: ``[time, channel]``. Channel should always be the last axis, time
        the second last.

    Epoched Data
        Epoched Data is often Continuous Data split into several equally long
        chunks (epochs). Each epoch usually belongs to a class. The axes in this
        case are ``[class, time, channel]``. Class should always be the first
        axis, time the second last and channel the last one. This is consistent
        with Continuos Data.

        Epoched Data can also contain different data than (e.g. data in the
        frequency domain), but the class axis should always be the first.

    Feature Vector
        In the later steps of the data processing, one often deals no more with
        continuous data but with feature vectors. Feature Vectors are similar to
        Epoched data, since each vector usually belongs to a class. Thus the
        axes are: ``[class, fv]``.

You are free to follow the convention or not. If you do, most methods will work
out of the box -- off course you still have to think if a certain method makes
sense on the current object at hand.

If you create non-conventional ``Data`` objects, the methods will still work (if
they make sense), but you have to provide the methods an extra parameter, with
the index of the axis (or axes).

Associating Samples to Timestamps
---------------------------------

The time marks the time at the *beginning* of the sample.

Example::

    Time  [ms]  0    10   20   30 ...
                |    |    |    |
    Sample [#]  [ 0 ][ 1 ][ 2 ][ 3 ]

The interpretation is that sample 0 contains the data from ``[0, 10)``, sample 1
contains ``[10, 20)``, and so on.


Intervals
---------

Whenever you encounter a time interval with a start and stop value, the
convention is ``[start, stop)`` (i.e. start is *included*, stop is *excluded*).


Example::

    Time  [ms]  0    10   20   30 ...
                |    |    |    |
    Sample [#]  [ 0 ][ 1 ][ 2 ][ 3 ]


Interval (0, 30) returns the samples 0, 1, 2

