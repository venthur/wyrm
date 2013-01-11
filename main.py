#!/usr/bin/env python


from os import path
import logging
import re

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import signal

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger('foo')


def load_brain_vision_data(vhdr):
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
    # some assumptions about the data...
    assert file_dict['Common Infos']['DataFormat'] == 'BINARY'
    assert file_dict['Common Infos']['DataOrientation'] == 'MULTIPLEXED'
    assert file_dict['Binary Infos']['BinaryFormat'] == 'INT_16'
    print fs, n_channels
    print marker_f
    print data_f
    print channels
    # load EEG data
    logger.debug('Loading EEG Data.')
    data = np.fromfile(data_f, np.int16)
    data = data.reshape(-1, n_channels)
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
                mrk.append([mrk_pos, mrk_descr])
    return data, mrk, channels, fs


def plot_channels(data, n_channels):
    ax = []
    for i in range(n_channels):
        if i == 0:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1)
        else:
            a = plt.subplot(10, n_channels / 10 + 1, i + 1, sharex=ax[0], sharey=ax[0])
        ax.append(a)
        a.plot(data[:, i])
        a.set_title(channels[i])


def segmentation(data, mrk, start, end):
    data2 = []
    for i in mrk:
        i_start, i_end = i-start, i+end
        chunk = data[i_start:i_end]
        data2.append(chunk)
    return np.array(data2)


def filter_bp(data, fs, low, high):
    # band pass filter the data
    fs_n = fs * 0.5
    #logger.debug('Calculating butter order...')
    #butter_ord, f_butter = signal.buttord(ws=[(low - .1) / fs_n, (high + .1) / fs_n],
    #                                      wp=[low / fs_n, high / fs_n],
    #                                      gpass=0.1,
    #                                      gstop=3.0
    #                                      )

    #logger.debug("{ord} {fbutter} {low} {high}".format(**{'ord': butter_ord,
    #                                                      'fbutter': f_butter,
    #                                                      'low': low / fs_n,
    #                                                      'high': high / fs_n}))
    butter_ord = 4
    b, a = signal.butter(butter_ord, [low / fs_n, high / fs_n], btype='band')
    return signal.lfilter(b, a, data, axis=0)


def calculate_csp(class1, class2):
    """Calculate the Common Spatial Pattern for... XXX

    Parameter:
        class: (channels, time, trials)

    See: http://en.wikipedia.org/wiki/Common_spatial_pattern

    Returns: the weight vector for the channels?

    """
    # sven's super simple matlab code
    # function [W, A, lambda] = my_csp(X1, X2)
    #     % compute covariance matrices of the two classes
    #     C1 = compute_Covariance_Matrix(X1);
    #     C2 = compute_Covariance_Matrix(X2);
    #     % solution of CSP objective via generalized eigenvalue problem
    #     [W, D] = eig(C1-C2, C1+C2);
    #     % make sure the eigenvalues and eigenvectors are sorted correctly
    #     [lambda, sort_idx] = sort(diag(D), 'descend');
    #     W = W(:,sort_idx);
    #     A = inv(W)';

    n_channels = class1.shape[2]
    # TODO compute x1, x2
    x1 = class1.swapaxes(1, 2)
    x1 = x1.reshape(-1, n_channels)
    x2 = class2.swapaxes(1, 2)
    x2 = x2.reshape(-1, n_channels)
    # compute covariance matrices of the two classes
    c1 = np.cov(x1.transpose())
    c2 = np.cov(x2.transpose())
    # solution of csp objective via generalized eigenvalue problem
    d, v = sp.linalg.eig(c1-c2, c1+c2)
    # make sure the eigenvalues and -vectors are correctly sorted
    indx = np.argsort(np.diag(d))
    # reverse
    indx = indx[::-1]
    d = d.take(indx)
    v = v.take(indx)
    a = sp.linalg.inv(v)
    return v, a, d


if __name__ == '__main__':
    # load data
    logger.debug('Loading raw data...')
    #raw_data, mrk, channels, fs = load_brain_vision_data('data/online_leitstand2011smaVPlab.vhdr')
    raw_data, mrk, channels, fs = load_brain_vision_data('data/OnlineTrainFileVPfaz.vhdr')
    n_channels = len(channels)
    # band pass filter the data
    data = filter_bp(raw_data, fs, 0.5, 30)
    # subsampling
    data = data[::10, :]
    del raw_data
    mrk = map(lambda x: [int(x[0]) / 10, x[1]], mrk)

    # segmentation
    std = [x for x, y in mrk if y in ['S  %i' % i for i in range(2, 7)]]
    dev = [x for x, y in mrk if y in ['S %i' % i for i in range(12, 17)]]

    # segmentation [-200 800]
    data_std = segmentation(data, std, 20, 80)
    std_avg = np.average(data_std, axis=0)
    data_dev = segmentation(data, dev, 20, 80)
    dev_avg = np.average(data_dev, axis=0)
    print dev_avg.shape


    # preliminary ml code from interactive console:
    from sklearn.lda import LDA
    train = [i.ravel() for i in data_std[:50]]
    train2 = [i.ravel() for i in data_dev[:50]]
    train = np.append(train, train2, 0)
    labels = [0]*50 + [1]*50
    clf.fit(train, labels)
    test_std = clf.predict([i.ravel() for i in data_std[50:]])
    test_dev = clf.predict([i.ravel() for i in data_dev[50:]])


    # plotting
    plot_channels(std_avg, n_channels)
    plot_channels(dev_avg, n_channels)

    #ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    #ax2.plot(c2)
    #plt.grid()
    #for pos, descr in mrk:
    #    if descr.startswith('S'):
    #        plt.axvline(x=pos, label=descr, color='g')
    #    elif descr.startswith('R'):
    #        plt.axvline(x=pos, label=descr, color='r')
    # draw spectrogram
    #a = plt.subplot(n_channels+1, 1, n_channels + 1)
    #a.specgram(c2, 1024, Fs = 1000)
    plt.show()
