#!/usr/bin/env python


"""Example Auditory ERP analysis."""


import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn.lda import LDA

from wyrm import misc
from wyrm import plot

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger('Auditory ERP')


if __name__ == '__main__':
    # load data
    logger.debug('Loading raw data...')
    #raw_data, mrk, channels, fs = load_brain_vision_data('data/online_leitstand2011smaVPlab.vhdr')
    raw_data, mrk, channels, fs = misc.load_brain_vision_data('data/OnlineTrainFileVPfaz.vhdr')
    n_channels = len(channels)

    # remove unneeded channels
    raw_data = np.delete(raw_data, [0, 3, 6], 1)
    channels.pop(6)
    channels.pop(3)
    channels.pop(0)
    n_channels = len(channels)

    # band pass filter the data
    data = misc.filter_bp(raw_data, fs, 8, 12)
    # subsampling
    data = data[::10, :]
    del raw_data
    mrk = map(lambda x: [int(x[0]) / 10, x[1]], mrk)

    # segmentation
    std = [x for x, y in mrk if y in ['S  %i' % i for i in range(2, 7)]]
    dev = [x for x, y in mrk if y in ['S %i' % i for i in range(12, 17)]]

    # preliminary ml code from interactive console:
    TEST_STD_I = 500
    TEST_DEV_I = 100

    # segmentation [-200 800]
    data_std = misc.segmentation(data, std, 25, 35)
    std_avg = np.average(data_std, axis=0)
    data_dev = misc.segmentation(data, dev, 25, 35)
    #dev_avg = np.average(data_dev[:TEST_STD_I], axis=0)
    w, a, d = misc.calculate_csp(data_std[:TEST_DEV_I], data_dev)
    w = w[:, (0, 1, -2, -1)]
    # plot the pattern
    # TODO: check if this is really the pattern
    ii = 0
    for i in 0, 1, -2, -1:
        ii += 1
        plt.subplot('22%d' % ii)
        plot.plot_scalp(a[:, i], channels)

    plt.show()

    # WRONG!
    # CSP is not for ERP data but motor imagery.
    # apply csp and log-var to data
    #data_std = [np.dot(i, w) for i in data_std]
    #data_std = np.array(data_std)
    data_std = np.var(data_std, 1)
    data_std = np.log(data_std)

    #data_dev = [np.dot(i, w) for i in data_dev]
    #data_dev = np.array(data_dev)
    data_dev = np.var(data_dev, 1)
    data_dev = np.log(data_dev)

    train = data_std[:TEST_STD_I]
    train2 = data_dev[:TEST_DEV_I]
    train = np.append(train, train2, 0)
    labels = [0]*TEST_STD_I + [1]*TEST_DEV_I
    clf = LDA()
    clf.fit(train, labels)

    test_std = clf.predict(data_std[TEST_STD_I:])
    test_std = np.abs(test_std - 1)
    test_dev = clf.predict(data_dev[TEST_DEV_I:])
    result = np.append(test_std, test_dev)
    print sum(result) / float(len(result))

    l = len(data_std[TEST_STD_I:])
    l2 = len(data_dev[TEST_DEV_I:])
    tmp = data_std[TEST_STD_I:]
    tmp2 = data_dev[TEST_DEV_I:]
    tmp = np.append(tmp, tmp2, 0)
    print "Mean LDA accuracy: ", clf.score(tmp, np.append(np.zeros(l), np.ones(l2)))


    # plotting
    #plot_channels(std_avg, n_channels)
    #plot_channels(dev_avg, n_channels)

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
    #plt.show()
