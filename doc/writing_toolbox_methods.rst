Writing Toolbox Methods
=======================

Methods Must not Modify Their Arguments
---------------------------------------

The convention for this toolbox is, that toolbox methods **must** not alter
their arguments. This is important as arguments to methods are passed by
*reference* in Python and changing an attribute of a *mutable* object in a
method which was passed as an argument, will automatically change the object
outside of this method too.

Example::

    >>> def do_something(arg):
    ...     arg['foo'] = 2

    >>> obj = {'foo' : 1, 'bar' : 2}
    >>> do_something(obj)
    >>> obj
    {'bar': 2, 'foo': 2}

Using :meth:`~wyrm.types.Data.copy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users rely on the methods to leave their arguments unmodified. To assist you
with that, the :class:`~wyrm.types.Data` object, provides a
:meth:`~wyrm.types.Data.copy` method which returns a deep copy of the object.
This method also allows to selectively overwrite or create attributes in the new
copy of the object.

Example::

    >>> def subsample(dat):
    ... # some calculations
    ... new_data = dat.data[::2]
    ... dat = dat.copy(data=new_data)
    ... return dat

Testing
~~~~~~~

To ensure that your new method does indeed not alter its arguments, you should
write an appropriate unit test. The test should look like this: 

1. copy the argument before passing it to the method to test
2. call the method to test
3. check if the copy of the argument and the argument are still equal

::

    def test_subsample_copy(self):
        """Subsample must not modify argument."""
        cpy = self.dat.copy()             # 1
        subsample(self.dat)               # 2
        self.assertEqual(cpy, self.dat)   # 3

Methods Must not Rely on a Specific Order of the Axes
-----------------------------------------------------

Although there is a convention on how to represent Feature Vectors, Continuous-,
and Epoched data, your methods must not rely on the specific order of the axes.
Instead, your method should be written in a way that the position is chooseable
as a parameter of your method. Furthermore those parameters should have default
values with the defaults being the values following the convention.

For example, let's assume the new method ``subsample``, which modifies data on
the time-axis of the argument. Usually the time-axis is the second last one in
Continuous- and Epoched data

We define our method with a default ``timeaxis`` parameter set to ``-2``::

    def subsample(dat, freq, timeaxis=-2):
        # do the subsampling
        ...

So we can call the method without specifying it when we have conventional data::

    dat = subsample(dat, 100)

or we call it specifying the time-axis on other data which follows not our
convention but sub sampling yields still a meaningful result::

    foo = subsample(foo, 100, timeaxis=7)

Off course writing your method this way is a bit more complicated, but nut very
much if you know how to index your arrays without the ``__getitem__`` or ``[]``
operator.

Assume you want to take every second value from the last axis of your data::

    d = np.arange(20).reshape(4,5)
    d = d[..., ::2]

How do you rewrite this in a way that the axis is arbitrary? One option is to
use :func:`numpy.take` which applies an array of indices on axis::

    # create an index array with indices of the elements in `timeaxis`
    idx = np.arange(d.shape[timeaxis])
    # take only every second (0, 2, 4, 6, ...)
    idx = idx[::2]
    # apply this index array on the last axis of d
    d = d.take(idx, timeaxis)

Be careful not to apply boolean indexing Arrays with :func:`numpy.take`, for
that use :func:`numpy.compress`, which does the same like `take` just with
boolean arrays.

Another way to achieve the same is to use :func:`slice` and create tuples for
indexing dynamically::

    idx = [slice(None) for i in d.ndims]
    idx[timeaxis] = slice(None, None, 2)
    # idx is now equivalent to [:, ::2]
    d = d[idx]

This is possible since ``a[:, ::2]`` is the same as 
``a[slice(None), slice(None, None, 2)]`` and the fact that ``a[x, y]`` is just
syntactic sugar for ``a[[x, y]]``.

Sometimes it might be necessary to insert a new axis in order to make numpy's
broadcasting work properly. For that use :func:`numpy.expand_dims`

Testing
~~~~~~~

To test if your method really works with nonstandard axes, you should write a
swapaxes-test in the unit test for your method. The test usually looks like
this:

1. swap axes of your data
2. apply your method to the swapped data
3. un-swap axes of the result
4. test if the result is equal to the result of applying your method to the
   original data

::

    def test_subsample_swapaxes(self):
        """subsample must work with nonstandard timeaxis."""
        dat = swapaxes(self.dat, 0, 1)        # 1
        dat = subsample(dat, 10, timeaxis=1)  # 2
        dat = swapaxes(dat, 0, 1)             # 3
        dat2 = subsample(self.dat, 10)
        self.assertEqual(dat, dat2)           # 4

