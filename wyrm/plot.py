#!/usr/bin/env python


from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy import interpolate

import tentensystem as tts


def plot_scalp(v, channel):
    """Plot the values v for channel `channel` on a scalp."""

    channelpos = [tts.channels[c] for c in channel]
    points = [_calculate_stereographic_projection(i) for i in channelpos]
    x = [i[0] for i in points]
    y = [i[1] for i in points]
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


def _calculate_stereographic_projection(p):
    """Calculate the stereographic projection.

    Given a unit sphere with radius `r = 1` and center at the origin. Project
    the point `p = (x, y, z)` from the sphere's South pole (0, 0, -1) on a
    plane on the sphere's North pole (0, 0, 1).

    The formula is:

        P' = P * (2r / (r + z))

    Arguments:
        p: The point to be projected in cartesian coordinates.

    Returns:
        [x, y]: the projected point on the plane.

    """
    # P' = P * (2r / r + z)
    mu = 1 / (1 + p[2])
    x = p[0] * mu
    y = p[1] * mu
    return x, y


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
