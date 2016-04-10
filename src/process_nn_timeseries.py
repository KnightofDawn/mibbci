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

    # Preliminary values
    freq_s_decimated = 64.0
    decimation_factor = params.FREQ_S_BDF / freq_s_decimated
    num_channels = params.NUM_CHANNELS_BDF
    window_size_decimated_in_samples = int(params.WINDOW_SIZE_RAW_SECS * freq_s_decimated)
    num_event_types = params.NUM_EVENT_TYPES_BDF
    M_fir = int(1.0 * window_size_decimated_in_samples)
    labels_names = ['btn dn']
    #labels_names = ['rh', 'lh', 'idle']

    # Parse command line
    #command_line_args_str = str(sys.argv)
    #print 'Command line args:', command_line_args_str
    #print 'sys.argv[1]:', sys.argv[1]
    if len(sys.argv) > 1:
        if sys.argv[1] == '-runtest':
            num_max_epochs = params.NUM_MAX_TRAIN_EPOCHS_RUNTEST
            num_train_data_instances = params.NUM_TRAIN_DATA_INSTANCES_RUNTEST
            print 'Main started in runtest mode.'
        else:
            print 'Error: Unrecognized command line arguments.'
            sys.exit()
    else:
        num_max_epochs = params.NUM_MAX_TRAIN_EPOCHS
        num_train_data_instances = params.NUM_TRAIN_DATA_INSTANCES
        print 'Main started in normal mode.'

    # Select the data files
    #data_filename_train_list = []
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h10m35s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h17m46s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h26m54s.csv')
    #data_filename_train_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h26m13s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_18h57m57s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h08m55s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h13m01s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h17m10s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h21m49s_RAW.csv')
    data_filename_train_list = []
    data_filename_train_list.append(
            'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\Emri_Akos_20160330\\emak_lt11.bdf')
    #        'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\Emri_Akos_20160330\\emak_rc11.bdf')

    # Load/pretrain the net
    is_net_pretrained = False
    #is_net_pretrained = True
    is_net_to_train_more = False
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
                    freq_s_decimated, M_fir)
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
                num_inputs=(num_channels, window_size_decimated_in_samples),
                num_outputs=num_event_types,
                num_max_epochs=num_max_epochs)

        # Load the training data
        X_train_raw, labels_train = utils.load_data(data_filename_train_list, decimation_factor)

        # Preprocess the data
        numer, denom, scaler = utils.init_preprocessors(
                X_train_raw,
                freq_s_decimated, M_fir)
        X_train_preproc, labels_train = utils.preprocess(
                X_train_raw, labels_train,
                tdfilt_numer=numer, tdfilt_denom=denom,
                # reref_channel_id=params.REREF_CHANNEL_ID,
                # power=True,
                # mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                scaler=scaler)
        # labels_train = labels_train

        # Train the NN
        nnet = nnutils.train_nn_from_timeseries(
                nnet,
                X_train_preproc, labels_train,
                window_size_decimated_in_samples,
                num_event_types,
                num_train_data_instances,
                plot_history=True)

        # Save the pipeline
        utils.save_processing_pipeline(nnet, numer, denom, scaler)
    print 'Training the NN finished.'

    # Load the test data
    print 'Loading the test data...'
    data_filename_test_list = []
    #data_filename_test_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h32m14s.csv')
    #data_filename_test_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h37m27s_RAW.csv')
    #data_filename_test_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h26m01s_RAW.csv')
    data_filename_test_list = []
    data_filename_test_list.append(
            'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\Emri_Akos_20160330\\emak_lt11.bdf')
    #        'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\Emri_Akos_20160330\\emak_rc11.bdf')
    X_test_raw, labels_test = utils.load_data(data_filename_test_list, decimation_factor)

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
    print TAG, 'Testing the net...'
    predictions = nnet.predict_proba(indices_test)
    print TAG, 'predictions size:', predictions.shape

    # Find the thresholds
    tpr_targets = (0.4, 0.4, 0.0)
    #p_thresholds = utils.calculate_auroc(labels_test, predictions, labels_names, tpr_targets, plot=True)
    utils.calculate_auroc(labels_test, predictions, labels_names, tpr_targets, plot=True)
    p_thresholds = (0.7, 0.7, 0.7)
    print 'p_thresholds:', p_thresholds

    #
    if False:
        create_control_signal(labels_test, predictions, p_thresholds);

    print 'Main terminates.'

