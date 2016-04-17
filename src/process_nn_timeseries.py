'''

TODO
- write a shell script which is in the root folder and calls src/....py
- auto save the roc images
- write net scheme to the param file somehow


http://rosinality.ncity.net/doku.php?id=python:installing_theano

'''

from timeseriesbatchiterator import TimeSeriesBatchIterator
import nnfactory
import nnutils
import utils
import params
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging


TAG = '[process_nn]'


########################################################################################################################

def create_control_signal(labels_test, predictions, p_thresholds):
    # Create the control signal time series
    # p_threshold_rh = 0.754
    # p_threshold_lh = 0.379
    predictions_rh = predictions[:, 0]
    predictions_lh = predictions[:, 1]
    predictions_rh_bin = predictions_rh
    predictions_lh_bin = predictions_lh
    predictions_rh_bin[predictions_rh_bin < p_thresholds[0]] = 0.0
    predictions_rh_bin[predictions_rh_bin >= p_thresholds[0]] = 1.0
    predictions_lh_bin[predictions_lh_bin < p_thresholds[1]] = 0.0
    predictions_lh_bin[predictions_lh_bin >= p_thresholds[1]] = 1.0
    predictions_res_bin = predictions_rh_bin - predictions_lh_bin
    labels_test_res = labels_test[:, 0] - labels_test[:, 1]

    # Create the cursor position time series
    cursor_pos_series = np.zeros(predictions_res_bin.shape)
    for i_time in range(1, predictions_res_bin.shape[0]):
        cursor_pos_series[i_time] = cursor_pos_series[i_time-1] + predictions_res_bin[i_time]

    # Plot prediction confidence histograms
    print 'predictions_rh:\n', predictions_rh
    plt.hist(x=predictions_rh, bins=100)
    plt.title("Right hand")
    plt.show()

    # Plot the timeline
    time_axis = np.arange(labels_test_res.shape[0])
    plt.plot(time_axis, labels_test_res, 'b-', label='event labels')
    plt.plot(time_axis, predictions_res_bin, 'rx', label='predictions')
    plt.plot(time_axis, 0.001*cursor_pos_series, 'r-', label='cursor x pos')
    #plt.plot(time_axis, labels_test[:, 0])
    #plt.plot(time_axis, predictions[:, 0])
    plt.legend(loc='lower right')
    plt.show()




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    # Init logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.debug('Main started.')
    #print 'Main started'

    # Parse command line
    #command_line_args_str = str(sys.argv)
    #print 'Command line args:', command_line_args_str
    #print 'sys.argv[1]:', sys.argv[1]
    is_runtest_mode = False
    if len(sys.argv) > 1:
        if sys.argv[1] == '--runtest':
            is_runtest_mode = True
            num_max_epochs = params.NUM_MAX_TRAIN_EPOCHS_RUNTEST
            num_train_data_instances = params.NUM_TRAIN_DATA_INSTANCES_RUNTEST
            #print 'Main started in runtest mode.'
            logging.debug('Main started in runtest mode.')
        else:
            #print 'Error: Unrecognized command line arguments.'
            logging.error('Error: Unrecognized command line arguments.')
            sys.exit()
    else:
        num_max_epochs = params.NUM_MAX_TRAIN_EPOCHS
        num_train_data_instances = params.NUM_TRAIN_DATA_INSTANCES
        logging.debug('Main started in normal mode.')
        #print 'Main started in normal mode.'

    # Preliminary values
    num_channels = 128
    num_event_types = 1
    freq_sampling = 128.0
    decimation_factor = 1.0
    freq_cut_lo = 4.0
    freq_cut_hi = 40.0
    window_size_decimated_in_samples = int(0.5 * freq_sampling)
    M_fir = int(1.0 * window_size_decimated_in_samples)
    labels_names = ['btn dn']
    #labels_names = ['rh', 'lh', 'idle']

    # Select the training data filenames
    data_filename_train_list = []
    data_filename_train_list.append('/home/user/Downloads/storage-double/OITI_2016/Pap_Henrik_20160404/pahe_rt11_128Hz.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h10m35s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h17m46s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h26m54s.csv')
    #data_filename_train_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h26m13s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_18h57m57s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h08m55s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h13m01s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h17m10s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h21m49s_RAW.csv')
    #data_filename_train_base = 'emak_rc11'
    #data_filename_train_base = 'pahe_rt11'
    #data_filename_train_list = []
    #data_filename_train_list.append(
    #        '/home/user/Downloads/storage-double/OITI_2016/Pap_Henrik_20160404/{}.bdf'.format(data_filename_train_base))
            #'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\Emri_Akos_20160330\\{}.bdf'.format(data_filename_train_base))

    # Select the test data filenames
    data_filename_test_list = []
    data_filename_test_list.append(data_filename_train_list[0])
    #data_filename_test_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h32m14s.csv')
    #data_filename_test_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h37m27s_RAW.csv')
    #data_filename_test_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h26m01s_RAW.csv')
    #data_filename_test_base = 'emak_rc11'
    #data_filename_test_base = 'pahe_rt11'
    #data_filename_test_list = []
    #data_filename_test_list.append(
    #        '/home/user/Downloads/storage-double/OITI_2016/Pap_Henrik_20160404/{}.bdf'.format(data_filename_test_base))
            #'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\Emri_Akos_20160330\\{}.bdf'.format(data_filename_test_base))

    # Set flow control switches
    is_net_pretrained = False
    #is_net_pretrained = True
    is_net_to_train_more = False
    is_data_plot_needed = True
    is_save_decimated_data_train_needed = True
    is_save_decimated_data_train_needed = False

    # Load/pretrain the net
    if is_net_pretrained:

        # Load the processing pipeline
        filename_pipeline_base = './models/MIBBCI_NN_20160406_14h17m59s'
        nnet, numer, denom, scaler = utils.load_processing_pipeline(filename_pipeline_base)

        if is_net_to_train_more:

            # Load the training data
            X_train_raw, labels_train = utils.load_data(data_filename_train_list, decimation_factor)

            # Preprocess the data
            numer, denom, scaler = utils.init_preprocessors(
                    X_train_raw,
                    freq_sampling,
                    freq_cut_lo,
                    freq_cut_hi,
                    M_fir)
            X_train_preproc, labels_train = utils.preprocess(
                X_train_raw, labels_train,
                tdfilt_numer=numer, tdfilt_denom=denom,
                # reref_channel_id=params.REREF_CHANNEL_ID,
                # power=True,
                # mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                scaler=scaler)
            # labels_train = labels_train

            # Train the NN
            nnet = nnfactory.train_nn_from_timeseries(
                    nnet,
                    X_train_preproc, labels_train,
                    window_size_decimated_in_samples,
                    num_event_types,
                    num_train_data_instances,
                    plot_history=True)

            # Save the pipeline
            utils.save_processing_pipeline(nnet, numer, denom, scaler)

    else:   # If a new net is to be created
        # Init the NN
        nnet, _ = nnfactory.create_nn_medium(
                num_inputs=(window_size_decimated_in_samples, num_channels),
                num_outputs=num_event_types,
                num_max_epochs=num_max_epochs)

        # Load the training data
        logging.debug('Loading the training data...')
        X_train_raw, labels_train = utils.load_data(
                data_filename_train_list,
                num_channels, num_event_types,
                decimation_factor)
        logging.debug('np.sum(labels_train): %f', np.sum(labels_train))
        logging.debug('Training data loaded.')


        # Preprocess the data
        numer, denom, scaler = utils.init_preprocessors(
                X_train_raw,
                freq_sampling,
                freq_cut_lo,
                freq_cut_hi,
                M_fir,
                plot=False)
        X_train_preproc, labels_train = utils.preprocess(
                X_train_raw, labels_train,
                tdfilt_numer=numer, tdfilt_denom=denom,
                # reref_channel_id=params.REREF_CHANNEL_ID,
                # power=True,
                # mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                scaler=scaler)
        # labels_train = labels_train

        # Plot the training data
        if is_data_plot_needed:
            logging.debug('Plotting the preprocessed training data...')
            time_axis = np.arange(X_train_preproc.shape[0])
            t_from = 10000
            t_to = 10512    #X_train_preproc.shape[0]
            channels_to_plot = (14, 29, 76, 112)
            #plt.plot(time_axis[t_from:t_to], X_train_raw[t_from:t_to, channels_to_plot], label='raw')
            plt.plot(time_axis[t_from:t_to], X_train_preproc[t_from:t_to, channels_to_plot], label='tdfilt')
            plt.plot(time_axis[t_from:t_to], 10.0*labels_train[t_from:t_to], label='event')
            plt.legend(loc='lower right')
            plt.show()

        # Train the NN
        logging.debug('Training the NN...')
        nnet = nnutils.train_nn_from_timeseries(
                nnet,
                X_train_preproc, labels_train,
                window_size_decimated_in_samples,
                num_event_types,
                num_train_data_instances,
                plot_history=True)
        logging.debug('Training the NN finished.')

        # Save the pipeline
        utils.save_processing_pipeline(nnet, numer, denom, scaler)


    # Load the test data
    logging.debug('Loading the test data...')
    X_test_raw, labels_test = utils.load_data(
            data_filename_test_list,
            num_channels, num_event_types,
            decimation_factor)
    if is_runtest_mode:
        X_test_raw = X_test_raw[0:1024]
        labels_test = labels_test[0:1024]

    # Pre-process the test data
    X_test_preproc, labels_test = utils.preprocess(
            X_test_raw, labels_test,
            tdfilt_numer=numer, tdfilt_denom=denom,
            #reref_channel_id=params.REREF_CHANNEL_ID,
            #power=True,
            #mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
            scaler=scaler)
    #labels_test = labels_test

    # Dummy set for testing
    #X_test_preproc = X_train = np.tile(np.reshape(labels_test[:, 0], (labels_test.shape[0], 1)), [1, params.NUM_CHANNELS])

    # Test the net
    batch_iter_test_valid = TimeSeriesBatchIterator(
            data=X_test_preproc, labels=None,
            window_size_samples=window_size_decimated_in_samples,
            num_outputs=num_event_types,
            batch_size=params.BATCH_SIZE)
    nnet.batch_iterator_train = None
    nnet.batch_iterator_test = batch_iter_test_valid
    indices_test = np.arange(X_test_preproc.shape[0])
    logging.debug('Testing the net...')
    utils.log_timestamp()
    predictions = nnet.predict_proba(indices_test)
    utils.log_timestamp()
    logging.debug('Predictions size: %d, %d', predictions.shape[0], predictions.shape[1])
    logging.debug('np.sum(predictions): %f', np.sum(predictions))

    # Find the thresholds
    tpr_targets = (0.4, 0.4, 0.0)
    #p_thresholds = utils.calculate_auroc(labels_test, predictions, labels_names, tpr_targets, plot=True)
    utils.calculate_auroc(labels_test, predictions, labels_names, tpr_targets, plot=True)
    p_thresholds = (0.7, 0.7, 0.7)
    logging.debug('p_thresholds: %f, %f, %f', p_thresholds[0], p_thresholds[1], p_thresholds[2])

    #
    if False:
        create_control_signal(labels_test, predictions, p_thresholds);

    logging.debug('Main terminates.')
