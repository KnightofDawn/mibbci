'''

    Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
        http://martinos.org/mne/dev/auto_examples/decoding/plot_decoding_csp_eeg.html


'''

import params
import numpy as np
from scipy import signal
from sklearn import linear_model
import matplotlib.pyplot as plt




########################################################################################################################


def process(X_in):
    X_preproc = X_in

    # Re-reference the data
    #id_ch_reref = 1    # FCz
    #X_rereferenced = X_in - np.tile(np.reshape(X_in[:, id_ch_reref], (X_in.shape[0], 1)), [1, params.NUM_CHANNELS]);
    X_rereferenced = X_in

    # Normalize the data
    #print 'X_preproc.mean(axis=0):', X_preproc.mean(axis=0)
    #print 'X_preproc.std(axis=0):', X_preproc.std(axis=0)
    #X_preproc = X_preproc - X_preproc.mean(axis=0)
    #X_preproc = X_preproc / X_preproc.std(axis=0)
    X_normalized = X_rereferenced
    print 'X_preproc.shape', X_preproc.shape

    # Initialize the time-domain filter
    freq_Nyq = params.FREQ_S/2.
    freq_trans = 0.5
    freqs_FIR_Hz = np.array([4.-freq_trans, 24.+freq_trans])
    #numer = signal.firwin(M_FIR, freqs_FIR, nyq=FREQ_S/2., pass_zero=False, window="hamming", scale=False)
    numer = signal.firwin(params.M_FIR, freqs_FIR_Hz, nyq=freq_Nyq, pass_zero=False, window="hamming", scale=False)
    denom = 1.
    '''w, h = signal.freqz(numer)
    plt.plot(freq_Nyq*w/math.pi, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()'''

    # Filter in time domain
    X_tdfilt = signal.lfilter(numer, denom, X_normalized.T).T
    print 'X_tdfilt.shape', X_tdfilt.shape

    # Get the signal power
    X_pow = X_tdfilt * X_tdfilt
    print 'X_pow.shape', X_pow.shape

    # Get moving average on signal power
    X_ma = signal.convolve(X_pow.T, np.ones((1, int(1*params.WINDOW_SIZE_RAW_SAMPLES)))).T
    X_ma = X_ma[0:X_pow.shape[0]]
    print 'X_ma.shape', X_ma.shape

    return X_ma





########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    print 'Main started.'

    # Load the training data
    label_id_red = 17
    data_filename_train_list = []
    data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h10m35s.csv')
    data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h17m46s.csv')
    data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h26m54s.csv')
    X_train_raw_list = []
    label_feed_train_list = []
    for data_filename in data_filename_train_list:
        data_loaded_train = np.loadtxt(fname=data_filename, delimiter=',', skiprows=1);
        print 'data_loaded.shape:', data_loaded_train.shape
        # TODO truncate
        X_train_raw_list.append(data_loaded_train[:, 1:(1 + params.NUM_CHANNELS)])
        label_feed_train_list.append(data_loaded_train[:, label_id_red])
        #print 'X_raw.shape', X_train_raw.shape
    X_train_raw = np.concatenate((X_train_raw_list[0], X_train_raw_list[1], X_train_raw_list[2]), axis=0)
    label_feed_train = np.concatenate((label_feed_train_list[0], label_feed_train_list[1], label_feed_train_list[2]), axis=0)

    # Process the data
    #print "X_raw[:, 1].shape:", X_raw[:, 1].shape
    #X_preproc = X_raw - np.tile(np.reshape(X_raw[:, 1], (X_raw.shape[0], 1)), [1, params.NUM_CHANNELS]);
    #X_preproc = X_raw
    X_train = process(X_train_raw)
    label_feed_train = label_feed_train
    #trunc_x_from = 2000
    #trunc_x_to = 10000
    #X_train = process(X_train_raw[trunc_x_from:trunc_x_to, :])
    #label_feed_train = label_feed_train[trunc_x_from:trunc_x_to]

    #time_axis = data_loaded_train[:, 0]

    # Init LogReg
    logreg_obj = linear_model.LogisticRegression()
    logreg_obj.fit(X_train, label_feed_train)
    print 'label_feed[trunc_x_from:trunc_x_to, 0].shape:', label_feed_train.shape

    # Load the test data
    data_filename_test = '../data/2016-03-26/MIBBCI_REC_20160326_15h32m14s.csv'
    data_loaded_test = np.loadtxt(fname=data_filename_test, delimiter=',', skiprows=1);
    print 'data_loaded.shape:', data_loaded_test.shape
    X_test_raw = data_loaded_test[:, 1:(1 + params.NUM_CHANNELS)]
    label_feed_test = data_loaded_test[:, label_id_red]
    print 'label_feed_test.shape:', label_feed_test.shape

    # Process the test data
    X_test = process(X_test_raw)

    # Classify the time instances
    #pred_feed = logreg_obj.predict(X_test)
    pred_feed = logreg_obj.predict(X_train)
    print 'pred_feed.shape:', pred_feed.shape

    # Get channel differences
    X_to_diff = X_train
    X_diff_c5_c6 = X_to_diff[:, 3] - X_to_diff[:, 9]
    X_diff_c3_c4 = X_to_diff[:, 4] - X_to_diff[:, 8]
    X_diff_c1_c2 = X_to_diff[:, 5] - X_to_diff[:, 7]
    X_diff_fc3_fc4 = X_to_diff[:, 0] - X_to_diff[:, 2]
    X_diff_cp5_cp6 = X_to_diff[:, 10] - X_to_diff[:, 12]
    X_diff_p3_p4 = X_to_diff[:, 13] - X_to_diff[:, 15]
    X_diff_avg = (X_diff_c5_c6 + X_diff_c3_c4 + X_diff_c1_c2 + X_diff_fc3_fc4 + X_diff_cp5_cp6 + X_diff_p3_p4) / 6.0


    # Plot the signal
    #if False:
    #    plt.plot(time_axis[trunc_x_from:trunc_x_to], 1.2*pred_feed)
    #    plt.plot(time_axis[trunc_x_from:trunc_x_to], label_feed_train[trunc_x_from:trunc_x_to, 0])
    #    plt.show()
    # Calculating AUROC

    if False:
        channels_to_plot = [1, 3, 9]
        #plt.plot(X_tdfilt[:, channels_to_plot])
        #plt.plot(X_ma[:, channels_to_plot])
        plt.plot(X_diff_c1_c2)
        plt.plot(X_diff_c3_c4)
        plt.plot(X_diff_c5_c6)
        plt.plot(X_diff_fc3_fc4)
        plt.plot(X_diff_cp5_cp6)
        plt.plot(X_diff_p3_p4)
        #plt.plot(X_diff_avg)
        plt.plot(1000 * label_feed_train[:, 0:2])
        plt.legend(['X_diff_c1_c2', 'X_diff_c3_c4', 'X_diff_c5_c6', 'X_diff_fc3_fc4', 'X_diff_cp5_cp6', 'X_diff_p3_p4', 'rh', 'lh'])
        plt.xlim([1500, 5000])
        plt.ylim([-2000, 2000])
        plt.show()
        #X_raw_mne.plot(events=events_mne, event_color={1: 'cyan'})
        #X_raw_mne.plot(events=event_series, event_color={1: 'cyan', -1: 'lightgray'})
        #time.sleep(2) no


