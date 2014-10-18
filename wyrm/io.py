# encoding: utf8

"""Various input/output methods.

This module provides methods for loading and saving data from- and into
various formats.

"""

from __future__ import division

from os import path
import logging
import re
import json
import socket

import numpy as np
from scipy.io import loadmat

from wyrm.types import Data


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


def save(dat, filename):
    """Save a ``Data`` object into a NumPy .npy file.


    Parameters
    ----------
    dat : Data
        `Data` object
    filename : str
        Filename of the file to save to. If the filename does not end
        with ``.npy``, the ``.npy`` extension will be automatically
        appended.


    See Also
    --------
    :func:`load`


    Examples
    --------

    >>> io.save(dat, 'foo.npy')
    >>> dat2 = io.load('foo.npy')

    """
    np.save(filename, dat)


def load(filename):
    """Load a ``Data`` object from a file.

    Parameters
    ----------
    filename : str
        the file to load the data from

    Returns
    -------
    dat : Data
        the data loaded from the file

    See Also
    --------
    :func:`save`

    Examples
    --------

    >>> io.save(dat, 'foo.npy')
    >>> dat2 = io.load('foo.npy')

    """
    dat = np.load(filename)
    # this is truly disgusting, apparently np.load returns a numpy array
    # that contains the pickled ``Data`` object but the array has no
    # elements, shape or whatever. The workaround is to flatten it and
    # return the first (i.e. only) element...
    dat = dat.flatten()[0]
    return dat


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


def load_mushu_data(meta):
    """Load saved EEG data in Mushu's format.

    This method loads saved data in Mushu's format and returns a
    continuous ``Data`` object.

    Parameters
    ----------
    meta : str
        Path to `.meta` file. A Mushu recording consists of three
        different files: `.eeg`, `.marker`, and `.meta`.

    Returns
    -------
    dat : Data
        Continuous Data object

    Examples
    --------

    >>> dat = load_mushu_data('testrecording.meta')

    """
    # reverse and replace and reverse again to replace only the last
    # (occurrence of .meta)
    datafile = meta[::-1].replace('atem.', 'gee.', 1)[::-1]
    markerfile = meta[::-1].replace('atem.', 'rekram.', 1)[::-1]
    assert path.exists(meta) and path.exists(datafile) and path.exists(markerfile)
    # load meta data
    with open(meta) as fh:
        metadata = json.load(fh)
    fs = metadata['Sampling Frequency']
    channels = np.array(metadata['Channels'])
    # load eeg data
    data = np.fromfile(datafile, np.float32)
    data = data.reshape((-1, len(channels)))
    # load markers
    markers = []
    with open(markerfile) as fh:
        for line in fh:
            ts, m = line.split(' ', 1)
            markers.append([float(ts), str(m).strip()])
    # construct Data
    duration = len(data) * 1000 / fs
    axes = [np.linspace(0, duration, len(data), endpoint=False), channels]
    names = ['time', 'channels']
    units = ['ms', '#']
    dat = Data(data=data, axes=axes, names=names, units=units)
    dat.fs = fs
    dat.markers = markers
    return dat


def convert_mushu_data(data, markers, fs, channels):
    """Convert mushu data into wyrm's ``Data`` format.

    This convenience method creates a continuous ``Data`` object from
    the parameters given. The timeaxis always starts from zero and its
    values are calculated from the sampling frequency ``fs`` and the
    length of ``data``. The ``names`` and ``units`` attributes are
    filled with default vaules.

    Parameters
    ----------
    data : 2d array
        an 2 dimensional numpy array with the axes: (time, channel)
    markers : list of tuples: (float, str)
        a list of markers. Each element is a tuple of timestamp and
        string. The timestamp is the time in ms relative to the onset of
        the block of data. Note that negative values are *allowed* as
        well as values bigger than the length of the block of data
        returned. That is to be interpreted as a marker from the last
        block and a marker for a future block respectively.
    fs : float
        the sampling frequency, this number is used to calculate the
        timeaxis for the data
    channels : list or 1d array of strings
        the channel names

    Returns
    -------
    cnt : continuous ``Data`` object

    Examples
    --------

    Assuming that ``amp`` is an Amplifier instance from ``libmushu``,
    already configured but not started yet:

    >>> amp_fs = amp.get_sampling_frequency()
    >>> amp_channels = amp.get_channels()
    >>> amp.start()
    >>> while True:
    ...     data, markers = amp.get_data()
    ...     cnt = convert_mushu_data(data, markers, amp_fs, amp_channels)
    ...     # some more code
    >>> amp.stop()

    References
    ----------
    https://github.com/venthur/mushu

    """
    time_axis = np.linspace(0, 1000 * data.shape[0] / fs, data.shape[0], endpoint=False)
    chan_axis = channels[:]
    axes = [time_axis, chan_axis]
    names = ['time', 'channel']
    units = ['uV', '#']
    cnt = Data(data=data.copy(), axes=axes, names=names, units=units)
    cnt.markers = markers[:]
    cnt.fs = fs
    return cnt


def load_bcicomp3_ds1(dirname):
    """Load the BCI Competition III Data Set 1.

    This method loads the data set and converts it into Wyrm's ``Data``
    format. Before you use it, you have to download the training- and
    test data in Matlab format and unpack it into a directory.

    .. note::

        If you need the true labels of the test sets, you'll have to
        download them separately from
        http://bbci.de/competition/iii/results/index.html#labels

    Parameters
    ----------
    dirname : str
        the directory where the ``Competition_train.mat`` and
        ``Competition_test.mat`` are located

    Returns
    -------
    epo_train, epo_test : epoched ``Data`` objects

    Examples
    --------

    >>> epo_test, epo_train = load_bcicomp3_ds1('/home/foo/bcicomp3_dataset1/')

    """
    # construct the filenames from the dirname
    training_file = path.sep.join([dirname, 'Competition_train.mat'])
    test_file = path.sep.join([dirname, 'Competition_test.mat'])

    # load the training data
    training_data_mat = loadmat(training_file)
    data = training_data_mat['X'].astype('double')
    data = data.swapaxes(-1, -2)
    labels = training_data_mat['Y'].astype('int').ravel()
    # convert into wyrm Data
    axes = [np.arange(i) for i in data.shape]
    axes[0] = labels
    axes[2] = [str(i) for i in range(data.shape[2])]
    names = ['Class', 'Time', 'Channel']
    units = ['#', 'ms', '#']
    dat_train = Data(data=data, axes=axes, names=names, units=units)
    dat_train.fs = 1000
    dat_train.class_names = ['pinky', 'tongue']

    # load the test data
    test_data_mat = loadmat(test_file)
    data = test_data_mat['X'].astype('double')
    data = data.swapaxes(-1, -2)
    # convert into wyrm Data
    axes = [np.arange(i) for i in data.shape]
    axes[2] = [str(i) for i in range(data.shape[2])]
    names = ['Epoch', 'Time', 'Channel']
    units = ['#', 'ms', '#']
    dat_test = Data(data=data, axes=axes, names=names, units=units)
    dat_test.fs = 1000

    # map labels -1 -> 0
    dat_test.axes[0][dat_test.axes[0] == -1] = 0
    dat_train.axes[0][dat_train.axes[0] == -1] = 0

    return dat_train, dat_test


def load_bcicomp3_ds2(filename):
    """Load the BCI Competition III Data Set 2.

    This method loads the data set and converts it into Wyrm's ``Data``
    format. Before you use it, you have to download the data set in
    Matlab format and unpack it. The directory with the extracted files
    must contain the ``Subject_*.mat``- and the ``eloc64.txt`` files.

    .. note::

        If you need the true labels of the test sets, you'll have to
        download them separately from
        http://bbci.de/competition/iii/results/index.html#labels

    Parameters
    ----------
    filename : str
        The path to the matlab file to load

    Returns
    -------
    cnt : continuous `Data` object


    Examples
    --------

    >>> dat = load_bcicomp3_ds2('/home/foo/data/Subject_A_Train.mat')

    """
    STIMULUS_CODE = {
        # cols from left to right
        1 : "agmsy5",
        2 : "bhntz6",
        3 : "ciou17",
        4 : "djpv28",
        5 : "ekqw39",
        6 : "flrx4_",
        # rows from top to bottom
        7 : "abcdef",
        8 : "ghijkl",
        9 : "mnopqr",
        10: "stuvwx",
        11: "yz1234",
        12: "56789_"
        }

    # load the matlab data
    data_mat = loadmat(filename)
    # load the channel names (the same for all datasets
    eloc_file = path.sep.join([path.dirname(filename), 'eloc64.txt'])
    with open(eloc_file) as fh:
        data = fh.read()
    channels = []
    for line in data.splitlines():
        if line:
            chan = line.split()[-1]
            chan = chan.replace('.', '')
            channels.append(chan)
    # fix the channel names, some letters have the wrong capitalization
    for i, s in enumerate(channels):
        s2 = s.upper()
        s2 = s2.replace('Z', 'z')
        s2 = s2.replace('FP', 'Fp')
        channels[i] = s2
    # The signal is recorded with 64 channels, bandpass filtered
    # 0.1-60Hz and digitized at 240Hz. The format is Character Epoch x
    # Samples x Channels
    data = data_mat['Signal']
    data = data.astype('double')
    # For each sample: 1 if a row/colum was flashed, 0 otherwise
    flashing = data_mat['Flashing'].reshape(-1)
    #flashing = np.flatnonzero((np.diff(a) == 1)) + 1
    tmp = []
    for i, _ in enumerate(flashing):
        if i == 0:
            tmp.append(flashing[i])
            continue
        if flashing[i] == flashing[i-1] == 1:
            tmp.append(0)
            continue
        tmp.append(flashing[i])
    flashing = np.array(tmp)
    # For each sample: 0 when no row/colum was intensified,
    # 1..6 for intensified columns, 7..12 for intensified rows
    stimulus_code = data_mat['StimulusCode'].reshape(-1)
    stimulus_code = stimulus_code[flashing == 1]
    # 0 if no row/col was intensified or the intensified did not contain
    # the target character, 1 otherwise
    stimulus_type = data_mat.get('StimulusType', np.array([])).reshape(-1)
    # The target characters
    target_chars = data_mat.get('TargetChar', np.array([])).reshape(-1)
    fs = 240
    data = data.reshape(-1, 64)
    timeaxis = np.linspace(0, data.shape[0] / fs * 1000, data.shape[0], endpoint=False)
    dat = Data(data=data, axes=[timeaxis, channels], names=['time', 'channel'], units=['ms', '#'])
    dat.fs = fs
    # preparing the markers
    target_mask = np.logical_and((flashing == 1), (stimulus_type == 1)) if len(stimulus_type) > 0 else []
    nontarget_mask = np.logical_and((flashing == 1), (stimulus_type == 0)) if len(stimulus_type) > 0 else []
    flashing = (flashing == 1)
    flashing = [[i, 'flashing'] for i in timeaxis[flashing]]
    targets = [[i, 'target'] for i in timeaxis[target_mask]]
    nontargets = [[i, 'nontarget'] for i in timeaxis[nontarget_mask]]
    dat.stimulus_code = stimulus_code[:]
    stimulus_code = zip([t for t, _ in flashing], [STIMULUS_CODE[i] for i in stimulus_code])
    markers = flashing[:]
    markers.extend(targets)
    markers.extend(nontargets)
    markers.extend(stimulus_code)
    markers.sort()
    dat.markers = markers[:]
    return dat


class PyffComm(object):
    """Pyff communication object.

    This class allows for communication with a running Pyff [1]_
    instance. It uses the json protocol, so you have to start Pyff with
    the ``--protocol=json`` parameter.

    Receiving data from Pyff (i.e. the available feedbacks and
    variables) is not supported for now.

    Examples
    --------

    This is an example session, demonstrating how to load a feedback
    application, set a variable, start it, quit it and closing Pyff in
    the end.

    >>> pyff = PyffComm()
    >>> pyff.send_init('TrivialPong')
    >>> pyff.set_variables({'FPS': 30})
    >>> pyff.play()
    >>> pyff.quit()
    >>> pyff.quit_pyff()

    References
    ----------

    .. [1] Bastian Venthur, Simon Scholler, John Williamson, Sven Dähne,
        Matthias S Treder, Maria T Kramarek, Klaus-Robert Müller and
        Benjamin Blankertz. Pyff---A Pythonic Framework for Feedback
        Applications and Stimulus Presentation in Neuroscience.
        Frontiers in Neuroscience. 2010. doi: 10.3389/fnins.2010.00179.

    """

    def __init__(self, host='localhost', port=12345):
        """Initialize the Pyff Communicator object.

        Parameters
        ----------
        host : str, optional
            hostname or IP of the machine running Pyff
        port : int, optional
            port on which Pyff is listening on.

        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = host
        self.port = port

    def send_interaction_signal(self, cmd, data=None):
        """Send interaction signal to Pyff.

        .. warning::
            This method is used internally to send low level JSON
            messages to Pyff. You should not use this method directly.

        Parameters
        ----------
        cmd : str
        data : dict

        """
        data = {'type' : 'interaction-signal',
                'data' : data,
                'commands' : [[cmd, dict()]]}
        json_str = json.dumps(data)
        self.socket.sendto(json_str, (self.host, self.port))

    # not sure if control signals are in use still, everything seems to
    # work with interaction signals as well
    #def send_control_signal(self, variables):
    #    data = {'type' : 'control-signal',
    #            'data' : variables,
    #            'commands' : None}
    #    json_str = json.dumps(data)
    #    self.send(json_str)

    def send_init(self, fb):
        """Load a Feedback.

        This method sends Pyff the ``send_init(feedback)`` command which
        loads a feedback.

        Parameters
        ----------
        fb : string
            The name of the feedback.

        """
        self.send_interaction_signal('sendinit', {'_feedback': fb})

    def play(self):
        """Start the feedback.

        """
        self.send_interaction_signal('play')

    def pause(self):
        """Pause the feedback.

        """
        self.send_interaction_signal('pause')

    def stop(self):
        """Stop the feedback.

        """
        self.send_interaction_signal('stop')

    def quit(self):
        """Quit the feedback.

        """
        self.send_interaction_signal('quit')

    def quit_pyff(self):
        """Quit Pyff.

        """
        self.send_interaction_signal('quitfeedbackcontroller')

    def set_variables(self, variables):
        """Set internal variables in the feedback.

        Use this method to create or modify instance variables of the
        currently running feedback.

        Parameters
        ----------
        variables : dict
            The variable names are the keys, the values are the values
            of the variables.

        """
        self.send_interaction_signal(None, variables)

