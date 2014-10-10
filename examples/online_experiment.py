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
from wyrm.types import RingBuffer
import wyrm.processing as proc
from wyrm import io


logging.basicConfig(format='%(relativeCreated)10.0f %(threadName)-10s %(name)-10s %(levelname)8s %(message)s', level=logging.NOTSET)
logger = logging.getLogger(__name__)

# replay the experiment in real time?
REALTIME = False


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

JUMPING_MEANS_IVALS = [150, 220], [200, 260], [310, 360], [550, 660] # 91%

SEG_IVAL = [0, 700]


def online_experiment(amp, cfy):
    amp_fs = amp.get_sampling_frequency()
    amp_channels = amp.get_channels()

    #buf = BlockBuffer(4)
    rb = RingBuffer(5000)

    fn = amp_fs / 2
    b_low, a_low = proc.signal.butter(5, [30 / fn], btype='low')
    b_high, a_high = proc.signal.butter(5, [.4 / fn], btype='high')

    zi_low = proc.lfilter_zi(b_low, a_low, len(amp_channels))
    zi_high = proc.lfilter_zi(b_high, a_high, len(amp_channels))

    amp.start()
    markers_processed = 0
    current_letter_idx = 0
    current_letter = TRUE_LABELS[current_letter_idx].lower()

    letter_prob = {i : 0 for i in 'abcdefghijklmnopqrstuvwxyz123456789_'}
    endresult = []
    t0 = time.time()
    while True:
        t0 = time.time()

        # get fresh data from the amp
        data, markers = amp.get_data()
        if len(data) == 0:
            continue

        # we should rather wait for a specific end-of-experiment marker
        if len(data) == 0:
            break

        # convert to cnt
        cnt = io.convert_mushu_data(data, markers, amp_fs, amp_channels)

        ## enter the block buffer
        #buf.append(cnt)
        #cnt = buf.get()
        #if not cnt:
        #    continue

        # band-pass and subsample
        cnt, zi_low = proc.lfilter(cnt, b_low, a_low, zi=zi_low)
        cnt, zi_high = proc.lfilter(cnt, b_high, a_high, zi=zi_high)

        cnt = proc.subsample(cnt, 60)

        newsamples = cnt.data.shape[0]

        # enter the ringbuffer
        rb.append(cnt)
        cnt = rb.get()

        # segment
        epo = proc.segment_dat(cnt, MARKER_DEF_TEST, SEG_IVAL, newsamples=newsamples)
        if not epo:
            continue

        fv = proc.jumping_means(epo, JUMPING_MEANS_IVALS)
        fv = proc.create_feature_vectors(fv)
        logger.debug(markers_processed)

        lda_out = proc.lda_apply(fv, cfy)
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
        # calculate the current accuracy
        if len(endresult) > 0:
            acc = np.count_nonzero(np.array(endresult) == np.array(list(TRUE_LABELS.lower()[:len(endresult)]))) / len(endresult)
            print "Current accuracy:", acc * 100
        if len(endresult) == len(TRUE_LABELS):
            break
        #logger.debug("Result: %s" % result)
        print 1000 * (time.time() - t0)

    acc = np.count_nonzero(np.array(endresult) == np.array(list(TRUE_LABELS.lower()[:len(endresult)]))) / len(endresult)
    print "Accuracy:", acc * 100

    amp.stop()


def train(filename):
    cnt = io.load_bcicomp3_ds2(filename)

    fs_n = cnt.fs / 2

    b, a = proc.signal.butter(5, [30 / fs_n], btype='low')
    cnt = proc.lfilter(cnt, b, a)

    b, a = proc.signal.butter(5, [.4 / fs_n], btype='high')
    cnt = proc.lfilter(cnt, b, a)

    cnt = proc.subsample(cnt, 60)

    epo = proc.segment_dat(cnt, MARKER_DEF_TRAIN, SEG_IVAL)

    #from wyrm import plot
    #plot.plot_spatio_temporal_r2_values(proc.sort_channels(epo))
    #print JUMPING_MEANS_IVALS
    #plot.plt.show()

    fv = proc.jumping_means(epo, JUMPING_MEANS_IVALS)
    fv = proc.create_feature_vectors(fv)

    cfy = proc.lda_train(fv)
    return cfy


if __name__ == '__main__':
    logger.debug('Training...')
    cfy = train(TRAIN_DATA)

    logger.debug('Starting Online experiment...')
    cnt = io.load_bcicomp3_ds2(TEST_DATA)
    amp = libmushu.get_amp('replayamp')
    if REALTIME:
        amp.configure(data=cnt.data, marker=cnt.markers, channels=cnt.axes[-1], fs=cnt.fs, blocksize_samples=4)
    else:
        amp.configure(data=cnt.data, marker=cnt.markers, channels=cnt.axes[-1], fs=cnt.fs, realtime=False, blocksize_samples=40)
    online_experiment(amp, cfy)

