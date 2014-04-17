#!/usr/bin/env python

"""Plotting methods.

This module contains various plotting methods. There are two types of
plotting methods: the Primitives and the Composites. The Primitives are
the most basic and offer simple, single-plot representations. The
Composites are composed of several primitives and offer more complex
representations.

.. warning::

    This module needs heavy reworking! We have yet to find a consistent
    way to handle primitive and composite plots, deal with the fact that
    some plots just manipulate axes, while others operate on figures and
    have to decide on which layer of matplotlib we want to deal with
    (i.e. pyplot, artist or even pylab).

    The API of this module will change and you should not rely on any
    method here.

"""


from __future__ import division

import math

import numpy as np
from scipy import interpolate
import matplotlib as mpl
from matplotlib import axes
from matplotlib import colorbar
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.path import Path
from matplotlib import patches as patches
from matplotlib.patches import Rectangle

from wyrm import processing as proc
from wyrm.types import Data


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



# ############# OLD FUNCTIONS ############################################


def plot_channels(dat, chanaxis=-1, otheraxis=-2):
    """Plot all channels for a continuous.

    Parameters
    ----------
    dat : Data

    """
    ax = []
    n_channels = dat.data.shape[chanaxis]
    for i, chan in enumerate(dat.axes[chanaxis]):
        if i == 0:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1)
        else:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1, sharex=ax[0], sharey=ax[0])
        ax.append(a)
        #x, y = dat.axes[otheraxis], dat.data.take([i], chanaxis)
        dat.axes[otheraxis], dat.data.take([i], chanaxis)
        a.plot(dat.axes[otheraxis], dat.data.take([i], chanaxis).squeeze())
        a.set_title(chan)
        plt.axvline(x=0)
        plt.axhline(y=0)


def plot_spatio_temporal_r2_values(dat):
    """Calculate the signed r^2 values and plot them in a heatmap.

    Parameters
    ----------
    dat : Data
        epoched data

    """
    r2 = proc.calculate_signed_r_square(dat)
    max = np.max(np.abs(r2))
    plt.imshow(r2.T, aspect='auto', interpolation='None', vmin=-max, vmax=max, cmap='RdBu')
    ax = plt.gca()
    # TODO: sort front-back, left-right
    # use the locators to fine-tune the ticks
    #ax.yaxis.set_major_locator(ticker.MaxNLocator())
    #ax.xaxis.set_major_locator(ticker.MaxNLocator())
    ax.yaxis.set_major_formatter(ticker.IndexFormatter(dat.axes[-1]))
    ax.xaxis.set_major_formatter(ticker.IndexFormatter(['%.1f' % i for i in dat.axes[-2]]))
    plt.xlabel('%s [%s]' % (dat.names[-2], dat.units[-2]))
    plt.ylabel('%s [%s]' % (dat.names[-1], dat.units[-1]))
    plt.tight_layout(True)
    plt.colorbar()
    plt.grid(True)


def plot_spectrum(spectrum, freqs):
    plt.plot(freqs, spectrum, '.')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('[dl]')


def plot_spectrogram(spectrogram, freqs):
    extent = 0, len(spectrogram), freqs[0], freqs[-1]
    plt.imshow(spectrogram.transpose(),
               aspect='auto',
               origin='lower',
               extent=extent,
               interpolation='none')
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time')


def interpolate_2d(x, y, z):
    """Interpolate missing points on a plane.

    Parameters
    ----------
    x, y, z : equally long lists of floats
        1d arrays defining points like ``p[x, y] = z``

    Returns
    -------
    X, Y, Z : 1d array, 1d array, 2d array
        ``Z`` is a 2d array ``[min(x)..max(x), [min(y)..max(y)]`` with
        the interpolated values as values.

    """
    xx = np.linspace(min(x), max(x))
    yy = np.linspace(min(y), max(y))
    xx, yy = np.meshgrid(xx, yy)
    f = interpolate.LinearNDInterpolator(zip(x, y), z)
    zz = f(xx, yy)
    return xx, yy, zz

# ############# COMPOSITE PLOTS ##########################################


def plot_timeinterval(data, r_square=None, highlights=None, hcolors=None,
                      legend=True, reg_chans=None, position=None):
    """Plots a simple time interval.

    Plots all channels of either continuous data or the mean of epoched
    data into a single timeinterval plot.

    Parameters
    ----------
    data : wyrm.types.Data
        Data object containing the data to plot.
    r_square : [values], optional
        List containing r_squared values to be plotted beneath the main
        plot (default: None).
    highlights : [[int, int)]
        List of tuples containing the start point (included) and end
        point (excluded) of each area to be highlighted (default: None).
    hcolors : [colors], optional
        A list of colors to use for the highlights areas (default:
        None).
    legend : Boolean, optional
        Flag to switch plotting of the legend on or off (default: True).
    reg_chans : [regular expression], optional
        A list of regular expressions. The plot will be limited to those
        channels matching the regular expressions. (default: None).
    position : [x, y, width, height], optional
        A Rectangle that limits the plot to its boundaries (default:
        None).

    Returns
    -------
    Matplotlib.Axes or (Matplotlib.Axes, Matplotlib.Axes)
        The Matplotlib.Axes corresponding to the plotted timeinterval
        and, if provided, the Axes corresponding to r_squared values.

    Examples
    --------
    Plots all channels contained in data with a legend.

    >>> plot_timeinterval(data)

    Same as above, but without the legend.

    >>> plot_timeinterval(data, legend=False)

    Adds r-square values to the plot.

    >>> plot_timeinterval(data, r_square=[values])

    Adds a highlighted area to the plot.

    >>> plot_timeinterval(data, highlights=[[200, 400]])

    To specify the colors of the highlighted areas use 'hcolors'.

    >>> plot_timeinterval(data, highlights=[[200, 400]], hcolors=['red'])
    """

    dcopy = data.copy()
    rect_ti_solo = [.07, .07, .9, .9]
    rect_ti_r2 = [.07, .12, .9, .85]
    rect_r2 = [.07, .07, .9, .05]

    if position is None:
        plt.figure()
        if r_square is None:
            pos_ti = rect_ti_solo
        else:
            pos_ti = rect_ti_r2
            pos_r2 = rect_r2
    else:
        if r_square is None:
            pos_ti = _transform_rect(position, rect_ti_solo)
        else:
            pos_ti = _transform_rect(position, rect_ti_r2)
            pos_r2 = _transform_rect(position, rect_r2)

    if reg_chans is not None:
        dcopy = proc.select_channels(dcopy, reg_chans)

    # process epoched data into continuous data using the mean
    if len(data.data.shape) > 2:
        dcopy = Data(np.mean(dcopy.data, axis=0), [dcopy.axes[-2], dcopy.axes[-1]],
                     [dcopy.names[-2], dcopy.names[-1]], [dcopy.units[-2], dcopy.units[-1]])

    ax1 = None
    # plotting of the data
    ax0 = _subplot_timeinterval(dcopy, position=pos_ti, epoch=-1, highlights=highlights,
                                hcolors=hcolors, legend=legend)
    ax0.xaxis.labelpad = 0
    if r_square is not None:
        ax1 = _subplot_r_square(r_square, position=pos_r2)
        ax0.tick_params(axis='x', direction='in', pad=30 * pos_ti[3])

    plt.grid(True)

    if r_square is None:
        return ax0
    else:
        return ax0, ax1


def plot_tenten(data, highlights=None, hcolors=None, legend=False, scale=True,
                reg_chans=None):
    """Plots channels on a grid system.

    Iterates over every channel in the data structure. If the
    channelname matches a channel in the tenten-system it will be
    plotted in a grid of rectangles. The grid is structured like the
    tenten-system itself, but in a simplified manner. The rows, in which
    channels appear, are predetermined, the channels are ordered
    automatically within their respective row. Areas to highlight can be
    specified, those areas will be marked with colors in every
    timeinterval plot.

    Parameters
    ----------
    data : wyrm.types.Data
        Data object containing the data to plot.
    highlights : [[int, int)]
        List of tuples containing the start point (included) and end
        point (excluded) of each area to be highlighted (default: None).
    hcolors : [colors], optional
        A list of colors to use for the highlight areas (default: None).
    legend : Boolean, optional
        Flag to switch plotting of the legend on or off (default: True).
    scale : Boolean, optional
        Flag to switch plotting of a scale in the top right corner of
        the grid (default: True)
    reg_chans : [regular expressions]
        A list of regular expressions. The plot will be limited to those
        channels matching the regular expressions.

    Returns
    -------
    [Matplotlib.Axes], Matplotlib.Axes
        Returns the plotted timeinterval axes as a list of
        Matplotlib.Axes and the plotted scale as a single
        Matplotlib.Axes.

    Examples
    --------
    Plotting of all channels within a Data object

    >>> plot_tenten(data)

    Plotting of all channels with a highlighted area

    >>> plot_tenten(data, highlights=[[200, 400]])

    Plotting of all channels beginning with 'A'

    >>> plot_tenten(data, reg_chans=['A.*'])
    """
    dcopy = data.copy()
    # this dictionary determines which y-position corresponds with which row in the grid
    ordering = {4.0: 0,
                3.5: 0,
                3.0: 1,
                2.5: 2,
                2.0: 3,
                1.5: 4,
                1.0: 5,
                0.5: 6,
                0.0: 7,
                -0.5: 8,
                -1.0: 9,
                -1.5: 10,
                -2.0: 11,
                -2.5: 12,
                -2.6: 12,
                -3.0: 13,
                -3.5: 14,
                -4.0: 15,
                -4.5: 15,
                -5.0: 16}

    # all the channels with their x- and y-position
    system = dict(CHANNEL_10_20)

    # create list with 17 empty lists. one for every potential row of channels.
    channel_lists = []
    for i in range(18):
        channel_lists.append([])

    if reg_chans is not None:
        dcopy = proc.select_channels(dcopy, reg_chans)

    # distribute the channels to the lists by their y-position
    count = 0
    for c in dcopy.axes[1]:
        if c in system:
            # entries in channel_lists: [<channel_name>, <x-position>, <position in Data>]
            channel_lists[ordering[system[c][1]]].append((c, system[c][0], count))
        count += 1

    # sort the lists of channels by their x-position
    for l in channel_lists:
        l.sort(key=lambda c_list: c_list[1])

    # calculate the needed dimensions of the grid
    columns = map(len, channel_lists)
    columns = [value for value in columns if value != 0]

    # add another axes to the first row for the scale
    columns[0] += 1

    plt.figure()
    grid = calc_centered_grid(columns, hpad=.01, vpad=.01)

    # axis used for sharing axes between channels
    masterax = None
    ax = []

    row = 0
    k = 0
    scale_ax = 0

    for l in channel_lists:
        if len(l) > 0:
            for i in range(len(l)):
                ax.append(_subplot_timeinterval(dcopy, grid[k], epoch=-1, highlights=highlights, hcolors=hcolors, labels=False,
                                                legend=legend, channel=l[i][2], shareaxis=masterax))
                if masterax is None and len(ax) > 0:
                    masterax = ax[0]

                # hide the axeslabeling
                plt.tick_params(axis='both', which='both', labelbottom='off', labeltop='off', labelleft='off',
                                labelright='off', top='off', right='off')

                # at this moment just to show what's what
                plt.gca().annotate(l[i][0], (0.05, 0.05), xycoords='axes fraction')

                k += 1

                if row == 0 and i == len(l)-1:
                    # this is the last axes in the first row
                    scale_ax = k
                    k += 1

            row += 1

    # plot the scale axes
    xtext = dcopy.axes[0][len(dcopy.axes[0])-1]
    sc = _subplot_scale(str(xtext) + ' ms', "$\mu$V", position=grid[scale_ax])

    return ax, sc


def plot_scalp(v, channels, levels=25, colormap=None, norm=None, ticks=None,
               annotate=True, position=None):
    """Plots the values 'v' for channels 'channels' on a scalp.

    Calculates the interpolation of the values v for the corresponding
    channels 'channels' and plots it as a contour plot on a scalp. The
    degree of gradients as well as the the appearance of the color bar
    can be adjusted.

    Parameters
    ----------
    v : [value]
        List containing the values of the channels.
    channels : [String]
        List containing the channel names.
    levels : int, optional
        The number of automatically created levels in the contour plot
        (default: 25).
    colormap : matplotlib.colors.colormap, optional
        A colormap to define the color transitions (default: a
        blue-white-red colormap).
    norm : matplotlib.colors.norm, optional
        A norm to define the min and max values (default: 'None', values
        from -10 to 10 are assumed).
    ticks : array([ints]), optional
        An array with values to define the ticks on the colorbar
        (default: 'None', 3 ticks at -10, 0 and 10 are displayed).
    annotate : Boolean, optional
        Flag to switch channel annotations on or off (default: True).
    position : [x, y, width, height], optional
        A Rectangle that limits the plot to its boundaries (default:
        None).

    Returns
    -------
    (Matplotlib.Axes, Matplotlib.Axes)
        Returns a pair of Matplotlib.Axes. The first contains the
        plotted scalp, the second the corresponding colorbar.

    Examples
    --------
    Plots the values v for channels 'channels' on a scalp

    >>> plot_scalp(v, channels)

    This plot has finer gradients through increasing the levels to 50.

    >>> plot_scalp(v, channels, levels=50)

    This plot has a norm and ticks from 0 to 10

    >>> n = matplotlib.colors.Normalize(vmin=0, vmax=10, clip=False)
    >>> t = np.linspace(0.0, 10.0, 3, endpoint=True)
    >>> plot_scalp(v, channels, norm=n, ticks=t)
    """
    rect_scalp = [.05, .05, .8, .9]
    rect_colorbar = [.9, .05, .05, .9]

    plt.figure(figsize=[8, 6.5])

    if position is None:
        pos_scalp = rect_scalp
        pos_colorbar = rect_colorbar
    else:
        pos_scalp = _transform_rect(position, rect_scalp)
        pos_colorbar = _transform_rect(position, rect_colorbar)

    if colormap is None:
        colormap = 'RdBu'
    if norm is None:
        norm = colors.Normalize(vmin=-10, vmax=10, clip=False)
    if ticks is None:
        ticks = np.linspace(-10.0, 10.0, 3, endpoint=True)

    ax0 = _subplot_scalp(v, channels, position=pos_scalp, levels=levels, colormap=colormap, annotate=annotate,
                         norm=norm)
    ax1 = _subplot_colorbar(position=pos_colorbar, colormap=colormap, ticks=ticks, norm=norm)

    return ax0, ax1


def plot_scalp_ti(v, channels, data, interval, scale_ti=.1, levels=25, colormap=None,
                  norm=None, ticks=None, annotate=True, position=None):
    """Plots a scalp with channels on top

    Plots the values v for channels 'channels' on a scalp as a contour
    plot. Additionaly plots the channels in channels_ti as a
    timeinterval on top of the scalp plot. The individual channels are
    placed over their position on the scalp.

    Parameters
    ----------
    v : [value]
        List containing the values of the channels.
    channels : [String]
        List containing the channel names.
    data : wyrm.types.Data
        Data object containing the continuous data for the overlaying
        timeinterval plots.
    interval : [begin, end)
        Tuple of ints to specify the range of the overlaying
        timeinterval plots.
    scale_ti : float, optional
        The percentage to scale the overlaying timeinterval plots
        (default: 0.1).
    levels : int, optional
        The number of automatically created levels in the contour plot
        (default: 25).
    colormap : matplotlib.colors.colormap, optional
        A colormap to define the color transitions (default: a
        blue-white-red colormap).
    norm : matplotlib.colors.norm, optional
        A norm to define the min and max values. If 'None', values from
        -10 to 10 are assumed (default: None).
    ticks : array([ints]), optional
        An array with values to define the ticks on the colorbar
        (default: None, 3  ticks at -10, 0 and 10 are displayed).
    annotate : Boolean, optional
        Flag to switch channel annotations on or off (default: True).
    position : [x, y, width, height], optional
        A Rectangle that limits the plot to its boundaries (default:
        None).

    Returns
    -------
    ((Matplotlib.Axes, Matplotlib.Axes), [Matplotlib.Axes])
        Returns a tuple of first a tuple with the plotted scalp and its
        colorbar, then a list of all on top plotted timeintervals.
    """
    rect_scalp = [.05, .05, .8, .9]
    rect_colorbar = [.9, .05, .05, .9]

    plt.figure(figsize=[16, 13])

    if position is None:
        pos_scalp = rect_scalp
        pos_colorbar = rect_colorbar
    else:
        pos_scalp = _transform_rect(position, rect_scalp)
        pos_colorbar = _transform_rect(position, rect_colorbar)

    if colormap is None:
        colormap = 'RdBu'
    if norm is None:
        norm = colors.Normalize(vmin=-10, vmax=10, clip=False)
    if ticks is None:
        ticks = np.linspace(-10.0, 10.0, 3, endpoint=True)

    ax0 = _subplot_scalp(v, channels, position=pos_scalp, levels=levels, annotate=annotate)
    ax1 = _subplot_colorbar(position=pos_colorbar, colormap=colormap, ticks=ticks, norm=norm)

    # modification of internally used data if a specific intervals is specified
    cdat = data.copy()
    if interval is not None:
        startindex = np.where(cdat.axes[0] == interval[0])[0][0]
        endindex = np.where(cdat.axes[0] == interval[1])[0][0]
        cdat.axes[0] = cdat.axes[0][startindex:endindex]
        cdat.data = cdat.data[startindex:endindex, :]

    tis = []
    for c in cdat.axes[1]:
        points = get_channelpos(c)
        if points is not None:
            channelindex = np.where(cdat.axes[1] == c)[0][0]

            # dirty: these are the x and y limits of the scalp axes
            minx = -1.15
            maxx = 1.15
            miny = -1.10
            maxy = 1.15

            # transformation of karth. to relative coordinates
            xy = (points[0] + (np.abs(minx))) * (1 / (np.abs(minx) + maxx)), \
                 (points[1] + (np.abs(miny))) * (1 / (np.abs(miny) + maxy))

            pos_c = [xy[0] - (scale_ti / 2), xy[1] - (scale_ti / 2), scale_ti, scale_ti]

            # transformation to fit into the scalp part of the plot
            pos_c = _transform_rect(pos_scalp, pos_c)

            tis.append(_subplot_timeinterval(cdat, position=pos_c, epoch=-1, highlights=None, legend=False,
                                             channel=channelindex, shareaxis=None))

        else:
            print('The channel "' + c + '" was not found in the tenten-system.')

    return (ax0, ax1), tis

# ############# TOOLS ####################################################


def set_highlights(highlights, hcolors=None, set_axes=None):
    """Sets highlights in form of vertical boxes to axes.

    Parameters
    ----------
    highlights : [(start, end)]
        List of tuples containing the start point (included) and end
        point (excluded) of each area to be highlighted.
    hcolors : [colors], optional
        A list of colors to use for the highlight areas (e.g. 'b',
        '#eeefff' or [R, G, B] for R, G, B = [0..1]. If left as None the
        colors blue, gree, red, cyan, magenta and yellow are used.
    set_axes : [matplotlib.axes.Axes], optional
        List of axes to highlights (default: None, all axes of the
        current figure will be highlighted).

    Examples
    ---------
    To create two highlighted areas in all axes of the currently active
    figure. The first area from 200ms - 300ms in blue and the second
    area from 500ms - 600ms in green.

    >>> set_highlights([[200, 300], [500, 600]])
    """
    if highlights is not None:

        if set_axes is None:
            set_axes = plt.gcf().axes

        def highlight(start, end, axis, color, alpha):
            axis.axvspan(start, end, edgecolor='w', facecolor=color, alpha=alpha)
            # the edges of the box are at the moment white. transparent edges
            # would be better.

        # create a standard variety of colors, if nothing is specified
        if hcolors is None:
            hcolors = ['b', 'g', 'r', 'c', 'm', 'y']

        # create a colormask containing #spans colors iterating over specified
        # colors or a standard variety
        colormask = []
        for index, span in enumerate(highlights):
            colormask.append(hcolors[index % len(hcolors)])

        # check if highlights is an instance of the Highlight class
        for p in set_axes:
            for span in highlights:
                highlight(span[0], span[1]-1, p, colormask.pop(0), .5)


def calc_grid(rows, cols, hpad=.05, vpad=.05):
    """ Calculates a grid of Rectangles and their positions.

    Parameters
    ----------
    rows : int
        The number of desired columns.
    cols : int
        The number of desired cols.
    hpad : float, optional
        The amount of horizontal padding (default: 0.05).
    vpad : float, optional
        The amount of vertical padding (default: 0.05).

    Returns
    -------
    [[float, float, float, float]]
        A list of all rectangle positions in the form of [xi, xy, width,
        height] sorted from top left to bottom right.

    Examples
    --------
    Calculates a 4x3 grid

    >>> calc_grid(4, 3)

    Calculates a 4x3 grid with more padding

    >>> calc_grid(4, 3, hpad=.1, vpad=.1)
    """
    w = (1 - ((cols + 1) * vpad)) / cols
    h = (1 - ((rows + 1) * hpad)) / rows

    grid = []
    for i in range(cols):
        for j in range(rows):
            xi = ((i % cols + 1) * hpad) + (i % cols * w)
            yj = 1 - (((j % rows + 1) * vpad) + ((j % rows + 1) * h))
            grid.append([xi, yj, w, h])

    return grid


def calc_centered_grid(cols_list, hpad=.05, vpad=.05):
    """Calculates a centered grid of Rectangles and their positions.

    Parameters
    ----------
    cols_list : [int]
        List of ints. Every entry represents a row with as many channels
        as the value.
    hpad : float, optional
        The amount of horizontal padding (default: 0.05).
    vpad : float, optional
        The amount of vertical padding (default: 0.05).

    Returns
    -------
    [[float, float, float, float]]
        A list of all rectangle positions in the form of [xi, xy, width,
        height] sorted from top left to bottom right.

    Examples
    --------
    Calculates a centered grid with 3 rows of 4, 3 and 2 columns

    >>> calc_centered_grid([4, 3, 2])

    Calculates a centered grid with more padding

    >>> calc_centered_grid([5, 4], hpad=.1, vpad=.75)
    """
    h = (1 - ((len(cols_list) + 1) * vpad)) / len(cols_list)
    w = (1 - ((max(cols_list) + 1) * hpad)) / max(cols_list)
    grid = []
    row = 1
    for l in cols_list:
        yi = 1 - ((row * vpad) + (row * h))
        for i in range(l):
            # calculate margin on both sides
            m = .5 - (((l * w) + ((l - 1) * hpad)) / 2)
            xi = m + (i * hpad) + (i * w)
            grid.append([xi, yi, w, h])
        row += 1
    return grid

# ############# PRIMITIVE PLOTS ##########################################


def _subplot_colorbar(position, colormap='RdBu', ticks=None, norm=None):
    """Creates a new axes with a colorbar.

    Creates a matplotlib.axes.Axes within the rectangle specified by
    'position' and fills it with a colorbar.

    Parameters
    ----------
    position : Rectangle
        The rectangle (x, y, width, height) where the axes will be
        created.
    colormap : matplotlib.colors.colormap, optional
        A colormap to define the colorscheme of the colormap.
    ticks : Array([float])
        An array with floats to set the number and location of the
        ticks.
    norm : matplotlib.colors.Normalize
        A norm to set the min-/max-value of the colorbar.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig = plt.gcf()
    ax = fig.add_axes(position)
    colorbar.ColorbarBase(ax, cmap=colormap, orientation='vertical', ticks=ticks, norm=norm)
    return ax


def _subplot_scalp(v, channels, position, levels=25, colormap=None, annotate=True, norm=None):
    """Creates a new axes with a scalp plot.

    Creates a matplotlib.axes.Axes within the rectangle specified by
    'position' and fills it with a contour plot for the channels in
    'channels' and the data in 'v'.

    Parameters
    ----------
    v : [value]
        List containing the values of the channels.
    channels : [String]
        List containing the channel names.
    position : Rectangle
        The rectangle (x, y, width, height) where the axes will be
        created.
    levels : int, optional
        The number of automatically created levels in the contour plot
        (default: 25).
    colormap : matplotlib.colors.colormap, optional
        A colormap to define the color transitions (default: a
        blue-white-red colormap).
    annotate : Boolean, optional
        Flag to switch channel annotations on or off (default: True).
    norm : matplotlib.colors.norm, optional
        A norm to define the min and max values (default: 'None', values
        from -10 to 10 are assumed).

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig = plt.gcf()
    ax = fig.add_axes(position)
    # what if we have an unknown channel?
    points = [get_channelpos(c) for c in channels]
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    z = v
    xx, yy, zz = interpolate_2d(x, y, z)

    if colormap is None:
        colormap = 'RdBu'

    ax.contourf(xx, yy, zz, levels, zorder=1, cmap=colormap, norm=norm)
    ax.contour(xx, yy, zz, levels, zorder=1, colors="k", norm=norm, linewidths=.1)

    ax.add_artist(plt.Circle((0, 0), radius=1, linewidth=3, fill=False))

    # add a nose
    ax.plot([-0.1, 0], [0.99, 1.1], 'k-', lw=2)
    ax.plot([0.1, 0], [0.99, 1.1], 'k-', lw=2)

    # add ears
    vertsr = [
        (0.99, 0.13),  # P0
        (1.10, 0.3),  # P1
        (1.10, -0.3),  # P2
        (0.99, -0.13)]  # P3

    vertsl = [
        (-0.99, 0.13),  # P0
        (-1.10, 0.3),  # P1
        (-1.10, -0.3),  # P2
        (-0.99, -0.13)]  # P3

    # in combination with Path this creates a bezier-curve with 2 fix-points and 2 control-points
    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4]

    pathr = Path(vertsr, codes)
    pathl = Path(vertsl, codes)
    patchr = patches.PathPatch(pathr, facecolor='none', lw=2)
    patchl = patches.PathPatch(pathl, facecolor='none', lw=2)
    ax.add_patch(patchr)
    ax.add_patch(patchl)

    # add markers at channels positions
    ax.plot(x, y, 'k+', ms=8, mew=1.2)

    # set the axes limits, so the figure is centered on the scalp
    ax.set_ylim([-1.05, 1.15])
    ax.set_xlim([-1.15, 1.15])

    # hide the axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if annotate:
        for i in zip(channels, zip(x, y)):
            ax.annotate(" " + i[0], i[1])

    return ax


def _subplot_timeinterval(data, position, epoch, highlights=None, hcolors=None,
                          labels=True, legend=True, channel=None, shareaxis=None):
    """Creates a new axes with a timeinterval plot.

    Creates a matplotlib.axes.Axes within the rectangle specified by
    'position' and fills it with a timeinterval plot defined by the
    channels and values contained in 'data'.

    Parameters
    ----------
    data : wyrm.types.Data
        Data object containing the data to plot.
    position : Rectangle
        The rectangle (x, y, width, height) where the axes will be
        created.
    epoch : int
        The epoch to be plotted. If there are no epochs this has to be
        '-1'.
    highlights : [[int, int)]
        List of tuples containing the start point (included) and end
        point (excluded) of each area to be highlighted (default: None).
    hcolors : [colors], optional
        A list of colors to use for the highlights areas (default:
        None).
    labels : Boolean, optional
        Flag to switch plotting of the usual labels on or off (default:
        True)
    legend : Boolean, optional
        Flag to switch plotting of the legend on or off (default: True).
    channel : int, optional
        This can be used to plot only a single channel. 'channel' has to
        be the index of the desired channel in data.axes[-1] (default:
        None)
    shareaxis : matplotlib.axes.Axes, optional
        An axes to share x- and y-axis with the new axes (default:
        None).

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig = plt.gcf()

    if shareaxis is None:
        ax = fig.add_axes(position)
    else:
        ax = axes.Axes(fig, position, sharex=shareaxis, sharey=shareaxis)
        fig.add_axes(ax)

    # epoch is -1 when there are no epochs
    if epoch == -1:
        if channel is None:
            ax.plot(data.axes[0], data.data)
        else:
            ax.plot(data.axes[0], data.data[:, channel])
    else:
        if channel is None:
            ax.plot(data.axes[len(data.axes) - 2], data.data[epoch])
        else:
            ax.plot(data.axes[len(data.axes) - 2], data.data[epoch, channel])

    # plotting of highlights
    if highlights is not None:
        set_highlights(highlights, hcolors=hcolors, set_axes=[ax])

    # labeling of axes
    if labels:
        ax.set_xlabel(data.units[0])
        ax.set_ylabel("$\mu$V")

    # labeling of channels
    if legend:
        if channel is None:
            ax.legend(data.axes[len(data.axes) - 1])
        else:
            ax.legend([data.axes[len(data.axes) - 1][channel]])

    ax.grid(True)
    return ax


def _subplot_r_square(data, position):
    """Creates a new axes with colored r-sqaure values.

    Parameters
    ----------
    data : [float]
        A list of floats that will be evenly distributed as colored
        tiles.
    position : Rectangle
        The rectangle (x, y, width, height) where the axes will be
        created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig = plt.gcf()
    ax = fig.add_axes(position)
    data = np.tile(data, (1, 1))
    ax.imshow(data, aspect='auto', interpolation='none')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def _subplot_scale(xvalue, yvalue, position):
    """Creates a new axes with a simple scale.

    Parameters
    ----------
    xvalue : String
        The text to be presented beneath the x-axis.
    yvalue : String
        The text to be presented next to the y-axis.
    position : Rectangle
        The rectangle (x, y, width, height) where the axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig = plt.gcf()
    ax = fig.add_axes(position)
    for item in [fig, ax]:
        item.patch.set_visible(False)
    ax.axis('off')
    ax.add_patch(Rectangle((1, 1), 3, .2, color='black'))
    ax.add_patch(Rectangle((1, 1), .1, 2, color='black'))
    plt.text(1.5, 2, yvalue)
    plt.text(1.5, .25, xvalue)
    ax.set_ylim([0, 4])
    ax.set_xlim([0, 5])
    return ax


def _transform_rect(rect, template):
    """Calculates the position of a relative notated rectangle within
    another rectangle.

    Parameters
    ----------
    rect : Rectangle
        The container rectangle to contain the other reactangle.
    template : Rectangle
        the rectangle to be contained in the other rectangle.
    """
    assert len(rect) == len(template) == 4, "Wrong inputs : [x, y, width, height]"
    x = rect[0] + (template[0] * rect[2])
    y = rect[1] + (template[1] * rect[3])
    w = rect[2] * template[2]
    h = rect[3] * template[3]
    return [x, y, w, h]


###############################################################################
# Utility Functions
###############################################################################

def get_channelpos(channame):
    """Return the x/y position of a channel.

    This method calculates the stereographic projection of a channel
    from ``CHANNEL_10_20``, suitable for a scalp plot.

    Parameters
    ----------
    channame : str
        Name of the channel, the search is case insensitive.

    Returns
    -------
    x, y : float or None
        The projected point on the plane if the point is known,
        otherwise ``None``

    Examples
    --------

    >>> plot.get_channelpos('C2')
    (0.1720792096741632, 0.0)
    >>> # the channels are case insensitive
    >>> plot.get_channelpos('c2')
    (0.1720792096741632, 0.0)
    >>> # lookup for an invalid channel
    >>> plot.get_channelpos('foo')
    None

    """
    channame = channame.lower()
    for i in CHANNEL_10_20:
        if i[0].lower() == channame:
            # convert the 90/4th angular position into x, y, z
            p = i[1]
            ea, eb = p[0] * (90 / 4), p[1] * (90 / 4)
            ea = ea * math.pi / 180
            eb = eb * math.pi / 180
            x = math.sin(ea) * math.cos(eb)
            y = math.sin(eb)
            z = math.cos(ea) * math.cos(eb)
            # Calculate the stereographic projection.
            # Given a unit sphere with radius ``r = 1`` and center at
            # the origin. Project the point ``p = (x, y, z)`` from the
            # sphere's South pole (0, 0, -1) on a plane on the sphere's
            # North pole (0, 0, 1).
            #
            # The formula is:
            #
            # P' = P * (2r / (r + z))
            #
            # We changed the values to move the point of projection
            # further below the south pole
            mu = 1 / (1.3 + z)
            x *= mu
            y *= mu
            return x, y
    return None


def beautify():
    """Set reasonable defaults matplotlib.

    This method replaces matplotlib's default rgb/cmyk colors with the
    colarized colors. It also does:

    * re-orders the default color cycle
    * sets the default linewidth
    * replaces the defaault 'RdBu' cmap
    * sets the default cmap to 'RdBu'

    Examples
    --------

    You can safely call ``beautify`` right after you've imported the
    ``plot`` module.

    >>> from wyrm import plot
    >>> plot.beautify()

    """
    def to_mpl_format(r, g, b):
        """Convert 0..255 t0 0..1."""
        return r / 256, g / 256, b / 256

    # The solarized color palette
    base03  = to_mpl_format(  0,  43,  54)
    base02  = to_mpl_format(  7,  54,  66)
    base01  = to_mpl_format( 88, 110, 117)
    base00  = to_mpl_format(101, 123, 131)
    base0   = to_mpl_format(131, 148, 150)
    base1   = to_mpl_format(147, 161, 161)
    base2   = to_mpl_format(238, 232, 213)
    base3   = to_mpl_format(253, 246, 227)
    yellow  = to_mpl_format(181, 137,   0)
    orange  = to_mpl_format(203,  75,  22)
    red     = to_mpl_format(220,  50,  47)
    magenta = to_mpl_format(211,  54, 130)
    violet  = to_mpl_format(108, 113, 196)
    blue    = to_mpl_format( 38, 139, 210)
    cyan    = to_mpl_format( 42, 161, 152)
    green   = to_mpl_format(133, 153,   0)

    white   = base3
    black   = base03

    # Tverwrite the default color values with our new ones. Those
    # single-letter colors are used all over the place in matplotlib, so
    # this setting has a huge effect.
    mpl.colors.ColorConverter.colors = {
        'b': blue,
        'c': cyan,
        'g': green,
        'k': black,
        'm': magenta,
        'r': red,
        'w': white,
        'y': yellow
    }

    # Redefine the existing 'RdBu' (Red-Blue) colormap, with our new
    # colors for red and blue
    cdict = {
        'red'  :  ((0., blue[0], blue[0]), (0.5, white[0], white[0]), (1., magenta[0], magenta[0])),
        'green':  ((0., blue[1], blue[1]), (0.5, white[1], white[1]), (1., magenta[1], magenta[1])),
        'blue' :  ((0., blue[2], blue[2]), (0.5, white[2], white[2]), (1., magenta[2], magenta[2]))
    }
    mpl.cm.register_cmap('RdBu', data=cdict)

    # Reorder the default color cycle
    mpl.rcParams['axes.color_cycle'] = ['b', 'm', 'g', 'r', 'c', 'y', 'k']
    # Set linewidth in plots to 2
    mpl.rcParams['lines.linewidth'] = 2
    # Set default cmap
    mpl.rcParams['image.cmap'] = 'RdBu'

