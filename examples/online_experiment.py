#!/usr/bin/env python


from __future__ import division

import sys
import time
import logging

import numpy as np

# change this to the path of mushu if you don't have it in your
# PYTHONPATH already
sys.path.append('../../mushu')
sys.path.append('../')

import libmushu
from wyrm.types import RingBuffer, BlockBuffer
import wyrm.processing as proc
from wyrm import io


logging.basicConfig(format='%(relativeCreated)10.0f %(threadName)-10s %(name)-10s %(levelname)8s %(message)s', level=logging.NOTSET)
logger = logging.getLogger(__name__)



TRAIN_DATA = 'data/BCI_Comp_III_Wads_2004/Subject_A_Train.mat'
TEST_DATA = 'data/BCI_Comp_III_Wads_2004/Subject_A_Test.mat'

CHANNEL_DATA = 'examples/data/BCI_Comp_III_Wads_2004/eloc64.txt'

TRUE_LABELS = "WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU"

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

MARKER_DEF_TRAIN = {'target': ['target'], 'nontarget': ['nontarget']}
MARKER_DEF_TEST = {i : [i] for i in STIMULUS_CODE.values()}
SEG_IVAL = [0, 600]


def online_experiment(amp, clf):
    amp_fs = amp.get_sampling_frequency()
    amp_channels = amp.get_channels()
    # TODO: check the correct block buffer size (and write it in the
    # docs dammit!
    # I assume it should be 12 = 240 / 20 (i.e. for subsampling)
    buf = BlockBuffer(50)
    rb = RingBuffer(2000)
    data = None
    markers = []
    fn = amp.get_sampling_frequency() / 2
    b, a = proc.signal.butter(5, [10 / fn], btype='low')
    filter_state = proc.signal.lfilter_zi(b, a)
    filter_state = np.array([filter_state for i in range(len(amp_channels))]).T
    amp.start()
    markers_processed = 0
    current_letter_idx = 0
    current_letter = TRUE_LABELS[current_letter_idx].lower()

    letter_prob = {i : 0 for i in 'abcdefghijklmnopqrstuvwxyz123456789_'}
    endresult = []
    while 1:
        time.sleep(0.01)
        # get fresh data from the amp
        data, markers = amp.get_data()

        # convert to cnt
        cnt = io.convert_mushu_data(data, markers, amp_fs, amp_channels)

        # enter the block buffer
        buf.append(cnt)
        cnt = buf.get()
        if not cnt:
            continue

        # band-pass and subsample
        cnt, filter_state = proc.lfilter(cnt, b, a, timeaxis=-2, zi=filter_state)
        cnt = proc.subsample(cnt, 20)
        newsamples = cnt.data.shape[0]

        # enter the ringbuffer
        rb.append(cnt)
        cnt = rb.get()

        # segment
        epo = proc.segment_dat(cnt, MARKER_DEF_TEST, SEG_IVAL, newsamples=newsamples)
        if not epo:
            continue
        fv = proc.create_feature_vectors(epo)
        logger.debug(markers_processed)

        lda_out = proc.lda_apply(fv, clf)
        markers = [fv.class_names[cls_idx] for cls_idx in fv.axes[0]]
        result = zip(markers, lda_out)
        for s, score in result:
            if markers_processed == 180:
                endresult.append(sorted(letter_prob.items(), key=lambda x: x[1])[-1][0])
                letter_prob = {i : 0 for i in 'abcdefghijklmnopqrstuvwxyz123456789_'}
                markers_processed = 0
                current_letter_idx += 1
                current_letter = TRUE_LABELS[current_letter_idx].lower()
            for letter in s:
                letter_prob[letter] += score
            markers_processed += 1
        logger.debug("".join([i[0] for i in sorted(letter_prob.items(), key=lambda x: x[1], reverse=True)]).replace(current_letter, " %s " % current_letter))
        logger.debug(TRUE_LABELS)
        logger.debug("".join(endresult))
        if len(endresult) == len(TRUE_LABELS):
            break
        #logger.debug("Result: %s" % result)

    endresult = np.array(endresult)
    true_labels = np.array(TRUE_LABELS)

    acc = np.count_nonzero(endresult == true_labels) / len(true_labels)
    print "Accuracy:", acc * 100

    amp.stop()


def train(filename):
    dat = io.load_bcicomp3_ds2(filename)
    b, a = proc.signal.butter(5, [10 / (dat.fs / 2)], btype='low')
    dat = proc.lfilter(dat, b, a)
    dat = proc.subsample(dat, 20)
    dat = proc.segment_dat(dat, MARKER_DEF_TRAIN, SEG_IVAL)
    dat = proc.create_feature_vectors(dat)
    clf = proc.lda_train(dat)
    return clf


if __name__ == '__main__':
    logger.debug('Training...')
    clf = train(TRAIN_DATA)

    logger.debug('Starting Online experiment...')
    cnt = io.load_bcicomp3_ds2(TEST_DATA)
    amp = libmushu.get_amp('replayamp')
    amp.configure(data=cnt.data, marker=cnt.markers, channels=cnt.axes[-1], fs=cnt.fs)
    online_experiment(amp, clf)

