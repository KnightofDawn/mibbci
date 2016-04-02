'''

    Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
        http://martinos.org/mne/dev/auto_examples/decoding/plot_decoding_csp_eeg.html


'''

import params
import math
import threading
import numpy as np
import time
from datetime import datetime
import csv
from scipy import signal
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import matplotlib.pyplot as plt




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    print 'Main started.'


    # Load the data
    data_filename = '../data/2016-03-26/MIBBCI_REC_20160326_15h10m35s.csv'
    data_loaded = np.loadtxt(fname=data_filename, delimiter=',', skiprows=1);
    print 'data_loaded.shape:', data_loaded.shape
    X_raw = data_loaded[:, 1:(1+params.NUM_CHANNELS)]
    time_axis = data_loaded[:, 0]
    label_feed = data_loaded[:, 17:20]
    print 'X_raw.shape', X_raw.shape

    # Preprocess the raw data
    trunc_x_from = 2000
    trunc_x_to = 10000
    X_preproc = X_raw[trunc_x_from:trunc_x_to, :]
    print 'X_preproc.mean(axis=0):', X_preproc.mean(axis=0)
    X_preproc = X_preproc - X_preproc.mean(axis=0)
    print 'X_preproc.std(axis=0):', X_preproc.std(axis=0)
    X_preproc = X_preproc / X_preproc.std(axis=0)

    # Initialize the time-domain filter
    freq_Nyq = params.FREQ_S/2.
    freq_trans = 0.5
    freqs_FIR_Hz = np.array([8.-freq_trans, 12.+freq_trans])
    #numer = signal.firwin(M_FIR, freqs_FIR, nyq=FREQ_S/2., pass_zero=False, window="hamming", scale=False)
    numer = signal.firwin(params.M_FIR, freqs_FIR_Hz, nyq=freq_Nyq, pass_zero=False, window="hamming", scale=False)
    denom = 1.
    '''w, h = signal.freqz(numer)
    plt.plot(freq_Nyq*w/math.pi, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()'''

    # Filter in time domain
    X_tdfilt = signal.lfilter(numer, denom, X_preproc.T).T

    # Get the signal power
    X_pow = X_tdfilt * X_tdfilt

    # Get moving average on signal power
    X_ma = signal.convolve(X_pow.T, np.ones((1, int(1*params.FREQ_S)))).T

    # Apply PCA
    pca_obj = PCA(n_components=params.NUM_CHANNELS)
    pca_obj.fit(X_ma)
    X_res = pca_obj.transform(X_ma)


    # Plot the signal
    if True:
        channels_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        plt.plot(X_res[:, channels_to_plot])
        plt.plot(label_feed[trunc_x_from:trunc_x_to, 0:2])
        #plt.xlim([1500, 5000])
        #plt.ylim([-2000, 2000])
        plt.show()


