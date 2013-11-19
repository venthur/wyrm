Plotting functions provided in this Toolbox //needs lot of work
===========================================

There are several plotting functions provided to create commonly used graphical representations of the data formats of this Toolbox.

Primitives
----------

These graphics are the most basic and offer simple, single-plot representations.

.. glossary::

    time interval
        A time interval plots continuous data of one or several channels to a time axis.

    scalp
        A scalp plot shows the data as it is spread about the subjects head.

    spectrogram
        No idea, yet.

Composites
----------

These graphics are composed of several primitives and offer more complex representations.

.. glossary::

    epoched time interval
        A plot of epoched data to represent the differences between different classes.

    ten ten system (time interval / scalp / spectrogram)
        A plot to show several channels sorted by their respective position on the scalp.

Plot-functions
--------------

These functions create a new figure and subplots according to their specifications.

Set-functions
-------------

These functions alter certain attributes of given subplots (matplotlib.Axes). These functions can be called after a plot-functions has been called to create an active plot.

Usage
-----


Examples
--------
