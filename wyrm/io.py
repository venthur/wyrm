
"""Input/Output methods.

This module provides methods for loading and saving data into various
formats.

"""

from __future__ import division

from os import path
import logging
import re

import numpy as np

from wyrm.types import Data


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)



def load_brain_vision_data(vhdr):
    """Load Brain Vision data from a file.

    This methods loads the continuous EEG data, and returns a ``Data``
    object of continuous data ``[time, channel]``, along with the
    markers and the sampling frequency. The EEG data is returned in
    micro Volt.

    Parameters
    ----------
    vhdr : str
        Path to a VHDR file

    Returns
    -------
    dat : Data
        Continuous Data with the additional attributes ``.fs`` for the
        sampling frequency and ``.marker`` for a list of markers. Each
        marker is a tuple of ``(time in ms, marker)``.

    Raises
    ------
    AssertionError
        If one of the consistency checks fails

    Examples
    --------

    >>> dat = load_brain_vision_data('path/to/vhdr')
    >>> dat.fs
    1000
    >>> dat.data.shape
    (54628, 61)

    """
    logger.debug('Loading Brain Vision Data Exchange Header File')
    with open(vhdr) as fh:
        fdata = map(str.strip, fh.readlines())
    fdata = filter(lambda x: not x.startswith(';'), fdata)
    fdata = filter(lambda x: len(x) > 0, fdata)
    # check for the correct file version:
    assert fdata[0].endswith('1.0')
    # read all data into a dict where the key is the stanza of the file
    file_dict = dict()
    for line in fdata[1:]:
        if line.startswith('[') and line.endswith(']'):
            current_stanza = line[1:-1]
            file_dict[current_stanza] = []
        else:
            file_dict[current_stanza].append(line)
    # translate known stanzas from simple list of strings to a dict
    for stanza in 'Common Infos', 'Binary Infos', 'Channel Infos':
        logger.debug(stanza)
        file_dict[stanza] = {line.split('=', 1)[0]: line.split('=', 1)[1] for line in file_dict[stanza]}
    # now file_dict contains the parsed data from the vhdr file
    # load the rest
    data_f = file_dict['Common Infos']['DataFile']
    marker_f = file_dict['Common Infos']['MarkerFile']
    data_f = path.sep.join([path.dirname(vhdr), data_f])
    marker_f = path.sep.join([path.dirname(vhdr), marker_f])
    n_channels = int(file_dict['Common Infos']['NumberOfChannels'])
    sampling_interval_microseconds = float(file_dict['Common Infos']['SamplingInterval'])
    fs = 1 / (sampling_interval_microseconds / 10**6)
    channels = [file_dict['Channel Infos']['Ch%i' % (i + 1)] for i in range(n_channels)]
    channels = map(lambda x: x.split(',')[0], channels)
    resolutions = [file_dict['Channel Infos']['Ch%i' % (i + 1)] for i in range(n_channels)]
    resolutions = map(lambda x: float(x.split(',')[2]), resolutions)
    # assert all channels have the same resolution of 0.1
    # FIXME: that is not always true, for example if we measure pulse or
    # emg
    #assert all([i == 0.1 for i in resolutions])
    # some assumptions about the data...
    assert file_dict['Common Infos']['DataFormat'] == 'BINARY'
    assert file_dict['Common Infos']['DataOrientation'] == 'MULTIPLEXED'
    assert file_dict['Binary Infos']['BinaryFormat'] == 'INT_16'
    # load EEG data
    logger.debug('Loading EEG Data.')
    data = np.fromfile(data_f, np.int16)
    data = data.reshape(-1, n_channels)
    data *= resolutions[0]
    n_samples = data.shape[0]
    # duration in ms
    duration = 1000 * n_samples / fs
    time = np.linspace(0, duration, n_samples, endpoint=False)
    # load marker
    logger.debug('Loading Marker.')
    regexp = r'^Mk(?P<mrk_nr>[0-9]*)=.*,(?P<mrk_descr>.*),(?P<mrk_pos>[0-9]*),[0-9]*,[0-9]*$'
    mrk = []
    with open(marker_f) as fh:
        for line in fh:
            line = line.strip()
            match = re.match(regexp, line)
            if match is None:
                continue
            mrk_pos = match.group('mrk_pos')
            mrk_descr = match.group('mrk_descr')
            if len(mrk_descr) > 1:
                # marker := [samplenr, marker]
                #mrk.append([int(mrk_pos), mrk_descr])
                # marker := [time in ms, marker]
                mrk.append([time[int(mrk_pos)], mrk_descr])
    dat = Data(data, [time, channels], ['time', 'channel'], ['ms', '#'])
    dat.fs = fs
    dat.markers = mrk
    return dat


