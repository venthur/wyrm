#!/usr/bin/env python


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy import interpolate

import tentensystem as tts


def plot_scalp(v, channel):
    """Plot the values v for channel `channel` on a scalp."""

    channelpos = [tts.channels[c] for c in channel]
    x = [-i/(-1.0-j) for i, _, j in channelpos]
    y = [-i/(-1.0-j) for _, i, j in channelpos]
    z = v
    X, Y, Z = _interpolate_2d(x, y, z)
    fig = plt.figure()
    #ax = plt.gca()
    ax = fig.add_subplot(111)
    im = ax.pcolor(X, Y, Z)
    patch = patches.Circle((0, 0), radius=1, facecolor='none', linewidth=3)
    ax.add_patch(patch)
    #im.set_clip_path(patch)
    #plt.contourf(X, Y, Z, alpha=.5)
    #ax.contourf(X, Y, Z)
    im = ax.contour(X, Y, Z)
    plt.clabel(im)
    #im.set_clip_path(patch)
    #plt.colorbar()
    plt.plot(x, y, 'bo')
    for i in zip(channel, zip(x,y)):
        plt.annotate(i[0], i[1])
    plt.show()


def _interpolate_2d(x, y, z):
    """Interpolate missing points on a plane.

    Arguments:
        x, y, z: 1d arrays defining points like p[x, y] = z

    Returns:
        X, Y, Z, where Z is a 2d array [min(x)..max(x), [min(y)..max(y)] with
        the interpolated values as values.

    """
    X = np.linspace(min(x), max(x))
    Y = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(X, Y)
    f = interpolate.interp2d(x, y, z)
    #f = interpolate.LinearNDInterpolator(zip(x, y), z)
    Z = f(X[0, :], Y[:, 0])
    return X, Y, Z
