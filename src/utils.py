

import nnutils
import params
import numpy as np
import sklearn
import scipy
import math
import matplotlib.pyplot as plt
import time
import datetime
import json
import cPickle
import pybdf
import joblib
import os
import logging
import matplotlib.animation as animation


TAG = '[utils]'


########################################################################################################################

def fit_scaler(X_raw):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_raw)

    return scaler

########################################################################################################################

def create_bandpass_filter(
        freq_cut_lo,
        freq_cut_hi,
        freq_s,
        M_fir,
        plot=False):

    # Initialize the time-domain filter
    freq_Nyq = freq_s / 2.
    #freqs_FIR_Hz = np.array([4. - freq_trans, 24. + freq_trans])
    freqs_FIR_Hz = np.array([freq_cut_lo, freq_cut_hi])
    print 'freq_Nyq:', freq_Nyq, 'freqs_FIR_Hz:', freqs_FIR_Hz
    # numer = scipy.signal.firwin(M_FIR, freqs_FIR, nyq=FREQ_S/2., pass_zero=False, window="hamming", scale=False)
    numer = scipy.signal.firwin(M_fir, freqs_FIR_Hz, nyq=freq_Nyq, pass_zero=False, window="hamming", scale=False)
    denom = np.array([1.])
    if plot:
        w, h = scipy.signal.freqz(numer)
        plt.plot(freq_Nyq*w/math.pi, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()

    return numer, denom


########################################################################################################################

def create_lowpass_filter(
        freq_cut,
        freq_s,
        M_fir,
        plot=False):

    # Initialize the time-domain filter
    freq_Nyq = freq_s / 2.
    numer = scipy.signal.firwin(numtaps=M_fir, cutoff=freq_cut, nyq=freq_Nyq)
    denom = np.array([1.])
    if plot:
        w, h = scipy.signal.freqz(numer)
        plt.plot(freq_Nyq*w/math.pi, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()

    return numer, denom

########################################################################################################################

def rereference(X_to_reref, id_ch_reref):
    X_rereferenced = X_to_reref - np.tile(np.reshape(X_to_reref[:, id_ch_reref], (X_to_reref.shape[0], 1)), [1, params.NUM_CHANNELS]);

    return X_rereferenced


########################################################################################################################

def init_preprocessors(
        X_raw,
        freq_sampling,
        freq_cut_lo,
        freq_cut_hi,
        M_fir,
        plot=False):

    # Init time-domain filters
    numer, denom = create_bandpass_filter(
            freq_cut_lo,
            freq_cut_hi,
            freq_sampling,
            M_fir,
            plot)
    print 'Created numer, denom:\n', numer, '\n', denom

    # Init scaler for the filtered data
    X_filt = scipy.signal.lfilter(numer, denom, X_raw.T).T
    scaler = fit_scaler(X_filt)
    print 'Fit scaler scaler.mean_, scaler.scale_:\n', scaler.mean_, '\n', scaler.scale_

    return numer, denom, scaler


########################################################################################################################

def preprocess(
        X_raw, labels,
        tdfilt_numer=None, tdfilt_denom=None,
        reref_channel_id=None,
        power=False, mov_avg_window_size=None,
        scaler=None):

    # Assign
    X_preprocessed = X_raw
    labels_preprocessed = labels

    # Time-domain filter
    if tdfilt_numer is None or tdfilt_denom is None:
        X_preprocessed = X_preprocessed
    else:
        X_preprocessed = scipy.signal.lfilter(tdfilt_numer, tdfilt_denom, X_preprocessed.T).T

    # Re-reference
    if reref_channel_id is not None:
        X_preprocessed = rereference(X_preprocessed, reref_channel_id)

    # Power
    if power is True:
        X_preprocessed = X_preprocessed * X_preprocessed

    # Moving average
    if mov_avg_window_size is not None:
        len_orig = X_preprocessed.shape[0]
        X_preprocessed = scipy.signal.convolve(X_preprocessed.T,
                np.ones((1, int(mov_avg_window_size * params.WINDOW_SIZE_DECIMATED_SAMPLES)))).T
        X_preprocessed = X_preprocessed[0:len_orig]

    # Scale
    X_preprocessed = scaler.transform(X_preprocessed)

    # Reshape
    X_preprocessed = flat_to_image(X_preprocessed, params.CHANNEL_NAMES_BS)

    # Plot
    #time_axis = np.arange(X_raw.shape[0])
    #len_plot = 1024
    #channels_to_plot = (4, 8)
    #labels_to_plot = 0
    ##plt.plot(time_axis[0:len_plot], X_raw[0:len_plot, channels_to_plot], label='Raw')
    #plt.plot(time_axis[0:len_plot], X_preprocessed[0:len_plot, channels_to_plot], label='Preproc')
    #plt.plot(time_axis[0:len_plot], labels[0:len_plot, labels_to_plot], label='Labels')
    #plt.legend(loc='lower right')
    #plt.show()

    return X_preprocessed, labels_preprocessed


########################################################################################################################


def calculate_auroc(
        labels_groundtruth,
        labels_predictions,
        labels_names,
        tpr_target_arr,
        plot=True):

    print 'labels_groundtruth.shape, labels_predictions.shape:', labels_groundtruth.shape, labels_predictions.shape

    fpr_arr_list = []
    tpr_arr_list = []
    threshold_arr_list = []
    auroc_val_list = []
    for i_event in range(labels_groundtruth.shape[1]):
        # fpr_arr, tpr_arr, _ = roc_curve(label_feed_test, pred_feed)
        fpr_arr_temp, tpr_arr_temp, threshold_arr_temp = sklearn.metrics.roc_curve(labels_groundtruth[:, i_event], labels_predictions[:, i_event])
        auroc_val = sklearn.metrics.auc(fpr_arr_temp, tpr_arr_temp)
        logging.debug('%s event %d auroc_val: %f', TAG, i_event, auroc_val)
        fpr_arr_list.append(fpr_arr_temp)
        tpr_arr_list.append(tpr_arr_temp)
        threshold_arr_list.append(threshold_arr_temp)
        auroc_val_list.append(auroc_val)

    # Get the threshold values for the acceptable TPR-s per event
    threshold_target_arr = np.zeros(labels_groundtruth.shape[1])
    print 'tpr_arr_list[0].shape:', tpr_arr_list[0].shape
    print 'labels_groundtruth.shape:', labels_groundtruth.shape
    for i_event in range(labels_groundtruth.shape[1]):
        for i_tpr in range(tpr_arr_list[i_event].shape[0] - 1):
            if tpr_target_arr[i_event] >= tpr_arr_list[i_event][i_tpr] and tpr_target_arr[i_event] < tpr_arr_list[i_event][i_tpr+1]:
                threshold_target_arr[i_event] = threshold_arr_list[i_event][i_tpr]

    if plot:
        # Plot TPR vs. thresholds
        for i_event in range(labels_groundtruth.shape[1]):
            plt.plot(threshold_arr_list[0], tpr_arr_list[0], label='tpr_{}'.format(i_event))
            plt.plot(threshold_arr_list[0], fpr_arr_list[0], label='fpr_{}'.format(i_event))
        #plt.plot(threshold_arr_list[0], tpr_arr_list[0], 'r-', label='rh - tpr')
        #plt.plot(threshold_arr_list[0], fpr_arr_list[0], 'r--', label='rh - fpr')
        #plt.plot(threshold_arr_list[1], tpr_arr_list[1], 'b-', label='lh - tpr')
        #plt.plot(threshold_arr_list[1], fpr_arr_list[1], 'b--', label='lh - fpr')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('TPR vs. thresholds')
        plt.legend(loc='upper right')
        plt.show()

    # Plot ROC
    for i_event in range(labels_groundtruth.shape[1]):
        plt.plot(fpr_arr_list[i_event], tpr_arr_list[i_event],
                 label='{0} (auc={1:0.4f})'.format(labels_names[i_event], auroc_val_list[i_event]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig('models/roc_{}.png'.format(datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR)), bbox_inches='tight')
    if plot:
        plt.show()

    return threshold_target_arr


########################################################################################################################

def create_epochs(X_timeseries, label_timeseries, time_offset_samples):
    X_epoch_list = []

    print 'X_timeseries.shape:', X_timeseries.shape
    print 'label_timeseries.shape:', label_timeseries.shape
    for i_time in range(1, X_timeseries.shape[0]):
        if label_timeseries[i_time-1] == 0.0 and label_timeseries[i_time] == 1.0:
            X_epoch_list.append(X_timeseries[
                    (i_time + time_offset_samples)
                    :(i_time + time_offset_samples + int(params.LEN_EPOCH_DECIMATED_SAMPLES)), :])

    return X_epoch_list


########################################################################################################################

def save_processing_pipeline(nn, nn_type, numer, denom, scaler):

    # Set the filename base
    time_save = datetime.datetime.now()
    filename_base = './models/MIBBCI_NN_{0}_{1}{2:02}{3:02}_{4:02}h{5:02}m{6:02}s'.format(
                    nn_type,
                    time_save.year, time_save.month, time_save.day,
                    time_save.hour, time_save.minute, time_save.second)

    # Save the NN
    filename_nn = filename_base + '.npz'
    nnutils.save_nn(nn, filename_nn)

    # Save the others to json
    print 'Saving numer, denom:\n', numer, '\n', denom
    print 'Saving scaler.mean_, scaler.scale_, scaler.var_:\n', scaler.mean_, '\n', scaler.scale_, '\n', scaler.var_
    indent = 4
    separators = (',', ': ')
    filename_json = filename_base + '.json'
    json_dict = {
            'nn_type': nn_type,
            'nn_filename': filename_nn,
            'numer': numer.tolist(),
            'denom': denom.tolist(),
            'scale_': scaler.scale_.tolist(),
            'mean_': scaler.mean_.tolist(),
            'var_': scaler.var_.tolist(),
            'n_samples_seen_': scaler.n_samples_seen_
            }
    with open(filename_json, 'w') as outfile:
        json.dump(json_dict, outfile, sort_keys=True, indent=indent, separators=separators)


    # Save the preproc stuff
    #print 'Before dump numer, denom:\n', numer, '\n', denom
    #filename_numer = filename_base + '_numer.p'
    #filename_denom = filename_base + '_denom.p'
    #cPickle.dump(numer, open(filename_numer, 'wb'))
    #cPickle.dump(numer, open(filename_denom, 'wb'))
    #print 'Before dump scaler.mean_, scaler.scale_:', scaler.mean_, scaler.scale_
    #filename_scaler = filename_base + '_scaler.p'
    #cPickle.dump(scaler, open(filename_scaler, 'wb'))

    # Test-load the NN with load_params_from
    # load_nn(create_nn_medium, filename_nn)

    # Test-load the preproc stuff
    # scaler = cPickle.load(open(filename_p, 'rb'))
    # print 'After load scaler.mean_, scaler.scale_:', scaler.mean_, scaler.scale_


########################################################################################################################

def load_processing_pipeline(
        filename_base,
        nn_type,
        num_inputs,
        num_outputs,
        num_max_training_epochs):

    # Load the others from json
    filename_json = filename_base + '.json'
    logging.debug('%s filename_json: %s', TAG, filename_json)
    with open(filename_json) as infile:
        json_root = json.load(infile)
    numer = np.asarray(json_root['numer'])
    denom = np.asarray(json_root['denom'])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.scale_ = np.asarray(json_root['scale_'])
    scaler.mean_ = np.asarray(json_root['mean_'])
    scaler.var_ = np.asarray(json_root['var_'])
    scaler.n_samples_seen_ = json_root['n_samples_seen_']
    nn_type = json_root['nn_type']
    filename_nn = json_root['nn_filename']
    logging.debug('%s nn_type, filename_nn: %s, %s', TAG, nn_type, filename_nn)
    print 'Loaded numer, denom:\n', numer, '\n', denom
    print 'Loaded scaler.mean_, scaler.scale_, scaler.var_:\n', scaler.mean_, '\n', scaler.scale_, '\n', scaler.var_

    # Load the nn
    filename_nn = filename_base + '.npz'
    nnet = nnutils.load_nn(
            filename_nn, nn_type,
            num_inputs, num_outputs,
            num_max_training_epochs)

    # Load the others
    #filename_numer = filename_base + '_numer.p'
    #filename_denom = filename_base + '_denom.p'
    #numer = cPickle.load(open(filename_numer, 'rb'))
    #denom = cPickle.load(open(filename_denom, 'rb'))
    #filename_scaler = filename_base + '_scaler.p'
    #scaler = cPickle.load(open(filename_scaler, 'rb'))
    #print 'Loaded numer, denom:\n', numer, '\n', denom
    #print 'Loaded scaler.mean_, scaler.scale_:\n', scaler.mean_, '\n', scaler.scale_

    return nnet, numer, denom, scaler


########################################################################################################################


def load_data(
    data_filename_list,
    num_channels,
    num_event_types,
    decimation_factor):

    # Find out the extensions
    # TODO decide extension per filename, not per the whole list
    filename = data_filename_list[0]
    file_extension = os.path.splitext(filename)[1]
    logging.debug('file_extension: %s', file_extension)

    # Call the appropriate load function depending on the file extension
    if file_extension == '.csv':
        X_raw, labels = load_data_csv(data_filename_list, num_channels, num_event_types, decimation_factor)
    elif file_extension == '.bdf':
        X_raw, labels = load_data_bdf(data_filename_list, decimation_factor)
    else:
        X_raw = None
        labels = None

    return X_raw, labels


########################################################################################################################

def load_data_csv(data_csv_filename_list, num_channels, num_event_types, decimation_factor):
    X_list = []
    labels_list = []

    for data_filename in data_csv_filename_list:
        logging.debug(data_filename)
        logging.debug('Loading data from %s ...', data_filename)
        data_loaded_train = np.loadtxt(fname=data_filename, delimiter=',', skiprows=1);
        print 'data_loaded.shape:', data_loaded_train.shape
        num_data_cols = data_loaded_train.shape[1]
        X_list.append(data_loaded_train[:, 1:(1 + num_channels)])
        labels_list.append(data_loaded_train[:, (num_data_cols - num_event_types):num_data_cols])
        # print 'X_raw.shape', X_train_raw.shape

    #X = np.concatenate((X_list[0], X_list[1], X_list[2]), axis=0)
    #labels = np.concatenate((labels_list[0], labels_list[1], labels_list[2]), axis=0)
    X_raw = np.concatenate(X_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Downsample the data
    downsample_indices = np.arange(0, X_raw.shape[0], int(decimation_factor)) + (int(decimation_factor)-1)
    print 'downsample_indices:\n', downsample_indices
    X_raw = X_raw[downsample_indices, 0:num_channels]
    labels = labels[downsample_indices]
    #X_raw = X_raw[::decimation_factor]
    #labels = labels[::decimation_factor]

    return X_raw, labels


########################################################################################################################


def load_data_bdf(data_bdf_filename_list, decimation_factor, num_cores=1):

    # TODO with list
    filename = data_bdf_filename_list[0]
    logging.debug('bdf filename: %s', filename)
    recording_obj = pybdf.bdfRecording(filename)

    # Get recording information
    logging.debug('sampling_rate [Hz]: %f', recording_obj.sampRate[0])
    logging.debug('duration [s]: %f', recording_obj.duration)
    logging.debug('#channels: %d', recording_obj.nChannels)
    logging.debug('unit of measure: %s', recording_obj.physDim[0])
    print 'channel labels:', recording_obj.chanLabels
    print 'dataChanLabels:', recording_obj.dataChanLabels

    # Convert the data channel labels to integers, otherwise pybdf runs into a bug
    for i_label in range(len(recording_obj.dataChanLabels)):
        recording_obj.dataChanLabels[i_label] = i_label
    print 'dataChanLabels:', recording_obj.dataChanLabels

    # Get the data object
    data_obj = recording_obj.getData()

    # Get the signal data from the data object
    logging.debug('Getting the signal data from the data object...')
    X_raw = data_obj['data'].T
    logging.debug('Getting the signal data from the data object finished.')
    logging.debug('X_raw shape: %d, %d', X_raw.shape[0], X_raw.shape[1])
    #time_axis = np.arange(X_raw.shape[1]) / recording_obj.sampRate[0]

    # Get the event information from the data object
    # Event codes: 247 is button pressed (1), 255 is button released (0)
    logging.debug('Getting the event data from the data object...')
    event_table = data_obj['eventTable']
    print 'event_table[\'code\']:\n', event_table['code']
    print 'event_table[\'idx\']:\n', event_table['idx']
    print 'event_table[\'dur\']:\n', event_table['dur']
    logging.debug('Getting the event data from the data object finished.')

    # Build the label feed from the event information
    logging.debug('Building the label feed from the event information...')
    labels = np.zeros((X_raw.shape[0], 1), np.float32)
    n_events = len(event_table['code'])
    EVENT_CODE_ACTION = 247
    #EVENT_CODE_IDLE = 255
    for i_event in range(1, n_events):
        if event_table['code'][i_event] == EVENT_CODE_ACTION:
            labels[(event_table['idx'][i_event] - params.EVENT_LENGTH_SAMPLES/2)
                    :(event_table['idx'][i_event] + params.EVENT_LENGTH_SAMPLES/2), :] = 1.0
        #if event_table['code'][i_event] == EVENT_CODE_ACTION:
        #    labels[event_table['idx'][i_event]:event_table['idx'][i_event+1], :] = 1.0
    logging.debug('np.sum(labels): %f', np.sum(labels))
    logging.debug('Building the label feed from the event information finished.')

    # Low-pass-filter the data before downsampling
    freq_sampling = recording_obj.sampRate[0]
    logging.debug('freq_sampling: %f', freq_sampling)
    freq_cut_downsampling = freq_sampling / (2.0 * decimation_factor)
    logging.debug('freq_cut_downsampling: %f', freq_cut_downsampling)
    M_fir = freq_sampling
    tdfilt_numer, tdfilt_denom = create_lowpass_filter(
            freq_cut=freq_cut_downsampling,
            freq_s=freq_sampling,
            M_fir=M_fir)
            #plot=True)
    logging.debug('Low-pass filtering the data before downsampling...')
    logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
    if num_cores > 1:
        #X_lpfiltered = np.array(joblib.Parallel(n_jobs=params.NUM_PARALLEL_JOBS)
        #                        (joblib.delayed(scipy.signal.lfilter)
        #                                (tdfilt_numer, tdfilt_denom,
        #                                X_raw[:, i_ch]) for i_ch in range(X_raw.shape[1]))).T
        X_lpfiltered = np.array(joblib.Parallel(n_jobs=params.NUM_PARALLEL_JOBS)
                                (joblib.delayed(scipy.signal.filtfilt)
                                        (tdfilt_numer, tdfilt_denom,
                                        X_raw[:, i_ch]) for i_ch in range(X_raw.shape[1]))).T
    else:
        #X_lpfiltered = scipy.signal.lfilter(tdfilt_numer, tdfilt_denom, X_raw.T).T
        X_lpfiltered = scipy.signal.filtfilt(tdfilt_numer, tdfilt_denom, X_raw.T).T
    logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
    logging.debug('Low-pass filtering the data before downsampling finished.')
    logging.debug('X_lpfiltered shape: %d, %d', X_lpfiltered.shape[0], X_lpfiltered.shape[1])

    # Plot the data
    if False:
        time_to = 2048*5
        time_axis = np.arange(time_to)
        channels_to_plot = (54, 111)
        plt.plot(time_axis, X_raw[0:time_to, channels_to_plot], label='raw')
        plt.plot(time_axis, X_lpfiltered[0:time_to, channels_to_plot], label='filt')
        #plt.plot(time_axis, labels[0:time_to], label='event')
        plt.legend(loc='lower right')
        plt.show()

    # Downsample the data
    downsample_indices = np.arange(0, X_lpfiltered.shape[0], int(decimation_factor)) + (int(decimation_factor)-1)
    print 'downsample_indices:\n', downsample_indices
    logging.debug('Downsampling the data...')
    logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
    X_downsampled = X_lpfiltered[downsample_indices, :]
    #X_downsampled = X_lpfiltered[downsample_indices, 0:params.NUM_CHANNELS_BDF]
    labels = labels[downsample_indices, :]
    #X_raw = X_raw[::decimation_factor]
    #labels = labels[::decimation_factor]
    logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
    logging.debug('Downsampling the data finished.')
    logging.debug('X_downsampled shape: %d, %d', X_downsampled.shape[0], X_downsampled.shape[1])

    return X_downsampled, labels


################################################################################

def log_timestamp():
    logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))


################################################################################

def flat_to_image(data, channel_names, plot=False):

    # Creating a dictionary
    channel_list = list(enumerate(channel_names))
    channel_id_dict = dict((channel[1], channel[0]) for channel in channel_list)
    print TAG, 'channel_id_dict:\n', channel_id_dict

    # Init the out data array
    data_images = np.zeros((data.shape[0],
            params.IMAGE_SIZE_BDF[0], params.IMAGE_SIZE_BDF[1]))
    logging.debug('%s data_images.shape: %d, %d, %d',
            TAG, data_images.shape[0], data_images.shape[1], data_images.shape[2])

    # Copy the channels
    for i_time in range(data.shape[0]):
        i_col = 0

        # col 0
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['A32']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['A31']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['A30']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['A29']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['A28']]
        data_images[i_time, 9, i_col] = data[i_time, channel_id_dict['A27']]
        data_images[i_time, 10, i_col] = data[i_time, channel_id_dict['A26']]
        i_col += 1
        # col 1
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['B5']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['A32']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['A32']]
        data_images[i_time, 9, i_col] = data[i_time, channel_id_dict['A32']]
        data_images[i_time, 10, i_col] = data[i_time, channel_id_dict['A32']]
        i_col += 1
        # col 2
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['B2']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['B3']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['B4']]
        i_col += 1
        # col 3
        data_images[i_time, 1, i_col] = data[i_time, channel_id_dict['B1']]
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['B19']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['B13']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['B12']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['B11']]
        data_images[i_time, 9, i_col] = data[i_time, channel_id_dict['B10']]
        i_col += 1
        # col 4
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['B18']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['B17']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['B16']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['B15']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['B14']]
        i_col += 1
        # col 5
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['B20']]
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['B21']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['B22']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['B23']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['B24']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['B25']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['B26']]
        i_col += 1
        # col 6
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['B31']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['B30']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['B29']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['B28']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['B27']]
        i_col += 1
        # col 7
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['B32']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['C5']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['C6']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['C7']]
        i_col += 1
        # col 8
        data_images[i_time, 1, i_col] = data[i_time, channel_id_dict['C1']]
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['C2']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['C3']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['C4']]
        i_col += 1
        # col 9
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['C11']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['C10']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['C9']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['C8']]
        i_col += 1
        # col 10
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['C12']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['C16']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['C13']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['C14']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['C15']]
        i_col += 1
        # col 11
        data_images[i_time, 0, i_col] = data[i_time, channel_id_dict['A1']]
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['C23']]
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['C22']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['C21']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['C20']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['C19']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['C18']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['C17']]
        i_col += 1
        # col 12
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['C25']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['C26']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['C27']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['C28']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['C29']]
        i_col += 1
        # col 13
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['C24']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['C32']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['C31']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['C30']]
        i_col += 1
        # col 14
        data_images[i_time, 1, i_col] = data[i_time, channel_id_dict['D1']]
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['D2']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['D3']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['D4']]
        i_col += 1
        # col 15
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['D13']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['D5']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['D6']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['D7']]
        i_col += 1
        # col 16
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['D12']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['D11']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['D10']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['D9']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['D8']]
        i_col += 1
        # col 17
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['D14']]
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['D18']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['D19']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['D20']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['D21']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['D22']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['D23']]
        i_col += 1
        # col 18
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['D28']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['D27']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['D26']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['D25']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['D24']]
        i_col += 1
        # col 19
        data_images[i_time, 1, i_col] = data[i_time, channel_id_dict['D15']]
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['D17']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['D29']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['D30']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['D31']]
        data_images[i_time, 9, i_col] = data[i_time, channel_id_dict['D32']]
        i_col += 1
        # col 20
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['D16']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['A6']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['A7']]
        i_col += 1
        # col 21
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['A8']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['A9']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['A10']]
        data_images[i_time, 9, i_col] = data[i_time, channel_id_dict['A11']]
        data_images[i_time, 10, i_col] = data[i_time, channel_id_dict['A12']]
        i_col += 1
        # col 22
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['A5']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['A18']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['A17']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['A16']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['A15']]
        data_images[i_time, 9, i_col] = data[i_time, channel_id_dict['A14']]
        data_images[i_time, 10, i_col] = data[i_time, channel_id_dict['A13']]
        i_col += 1
        # col 23
        data_images[i_time, 1, i_col] = data[i_time, channel_id_dict['A2']]
        data_images[i_time, 2, i_col] = data[i_time, channel_id_dict['A3']]
        data_images[i_time, 3, i_col] = data[i_time, channel_id_dict['A4']]
        data_images[i_time, 4, i_col] = data[i_time, channel_id_dict['A19']]
        data_images[i_time, 5, i_col] = data[i_time, channel_id_dict['A20']]
        data_images[i_time, 6, i_col] = data[i_time, channel_id_dict['A21']]
        data_images[i_time, 7, i_col] = data[i_time, channel_id_dict['A22']]
        data_images[i_time, 8, i_col] = data[i_time, channel_id_dict['A23']]
        data_images[i_time, 9, i_col] = data[i_time, channel_id_dict['A24']]
        data_images[i_time, 10, i_col] = data[i_time, channel_id_dict['A25']]
        i_col += 1

    # Plot test
    if plot:
        def generate_data(i_time):
            return data_images[i_time]
        def update(data):
            mat.set_data(data)
            return mat
        def data_gen():
            for i_time in range(data_images.shape[0]):
                yield generate_data(i_time)
        fig, ax = plt.subplots()
        mat = ax.matshow(generate_data(0))
        plt.colorbar(mat)
        ani = animation.FuncAnimation(fig, update, data_gen, interval=100)   #, save_count=50)
        plt.show()

    # Check an image
    if False:
        plt.matshow(data_images[-1])
        plt.show()

    return data_images

################################################################################
