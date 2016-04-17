'''

TODO
- auto save the roc images
- write net scheme to the param file somehow


http://rosinality.ncity.net/doku.php?id=python:installing_theano

'''

from epochbatchiterator import EpochBatchIterator
import params
import nnutils
import utils
import numpy as np
import matplotlib.pyplot as plt
import sys


TAG = '[process_nn]'




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

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
    data_filename_train_list = []
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h10m35s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h17m46s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h26m54s.csv')
    #data_filename_train_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h26m13s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_18h57m57s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h08m55s_RAW.csv')
    #data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h13m01s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h17m10s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h21m49s_RAW.csv')

    # Load/pretrain the net
    is_net_pretrained = False
    #is_net_pretrained = True
    is_net_to_train_more = False
    if is_net_pretrained:

        # Load the pipeline
        filename_base = '../models/MIBBCI_NN_20160406_14h17m59s'
        nnet, numer, denom, scaler = utils.load_pipeline(filename_base)

        if is_net_to_train_more:

            # Load the training data
            X_train_raw, labels_train = utils.load_data(data_filename_train_list)

            # Preprocess the data
            numer, denom, scaler = utils.init_preprocessors(X_train_raw)
            X_train_preproc, labels_train = utils.preprocess(
                X_train_raw, labels_train,
                decimation_factor=params.DECIMATION_FACTOR_PREPROC,
                tdfilt_numer=numer, tdfilt_denom=denom,
                # reref_channel_id=params.REREF_CHANNEL_ID,
                # power=True,
                # mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                scaler=scaler)
            # labels_train = labels_train

            # Epoch the data
            X_epoch_list_train_rh = utils.create_epochs(
                    X_train_preproc, labels_train[:, 0], params.EPOCH_OFFSET_SAMPLES)
            X_epoch_list_train_lh = utils.create_epochs(
                    X_train_preproc, labels_train[:, 1], params.EPOCH_OFFSET_SAMPLES)
            X_epoch_list_train_all = []
            X_epoch_list_train_all.extend(X_epoch_list_train_rh)
            X_epoch_list_train_all.extend(X_epoch_list_train_lh)
            label_list_train_all = []
            label_list_train_all.extend([1.0, 0.0] * len(X_epoch_list_train_rh))
            label_list_train_all.extend([0.0, 1.0] * len(X_epoch_list_train_lh))

            # Train the NN
            nnet = nnutils.train_nn_epochs(
                    nnet, X_epoch_list_train_all, label_list_train_all,
                    num_train_data_instances, plot_history=True)
            #nnet, numer, denom, scaler = nnutils.train_nn_timeseries(nnet, X_train_raw, labels_train, plot_history=True)

            # Save the pipeline
            utils.save_pipeline(nnet, numer, denom, scaler)

    else:   # If a new net is to be created
        # Init the NN
        nnet, _ = nnutils.create_nn_medium(
                num_inputs=(params.NUM_CHANNELS, params.LEN_EPOCH_DECIMATED_SAMPLES),
                num_outputs=2,
                num_max_epochs=num_max_epochs)

        # Load the training data
        X_train_raw, labels_train = utils.load_data(data_filename_train_list)

        # Preprocess the data
        numer, denom, scaler = utils.init_preprocessors(X_train_raw)
        X_train_preproc, labels_train = utils.preprocess(
                X_train_raw, labels_train,
                decimation_factor=params.DECIMATION_FACTOR_PREPROC,
                tdfilt_numer=numer, tdfilt_denom=denom,
                # reref_channel_id=params.REREF_CHANNEL_ID,
                # power=True,
                # mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                scaler=scaler)
        # labels_train = labels_train

        # Epoch the training data
        print 'X_train_preproc.shape:', X_train_preproc.shape
        X_epoch_list_train_rh = utils.create_epochs(
                X_train_preproc, labels_train[:, 0], params.EPOCH_OFFSET_SAMPLES)
        X_epoch_list_train_lh = utils.create_epochs(
                X_train_preproc, labels_train[:, 1], params.EPOCH_OFFSET_SAMPLES)
        X_epoch_list_train_all = []
        X_epoch_list_train_all.extend(X_epoch_list_train_rh)
        X_epoch_list_train_all.extend(X_epoch_list_train_lh)
        label_list_train_all = []
        label_list_train_all.extend([[1.0, 0.0]] * len(X_epoch_list_train_rh))
        label_list_train_all.extend([[0.0, 1.0]] * len(X_epoch_list_train_lh))

        # Train the NN
        #print 'X_epoch_list_all[0].shape:', X_epoch_list_train_all[0].shape
        nnet = nnutils.train_nn_epochs(
                nnet, X_epoch_list_train_all, label_list_train_all,
                num_train_data_instances, plot_history=True)
        #nnet, numer, denom, scaler = nnutils.train_nn_timeseries(nnet, X_train_raw, labels_train, plot_history=True)

        # Save the pipeline
        utils.save_pipeline(nnet, numer, denom, scaler)

    # Load the test data
    data_filename_test_list = []
    #data_filename_test_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h32m14s.csv')
    #data_filename_test_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h37m27s_RAW.csv')
    data_filename_test_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h26m01s_RAW.csv')
    X_test_raw, labels_test = utils.load_data(data_filename_test_list)

    # Pre-process the test data
    X_test_preproc, labels_test = utils.preprocess(
            X_test_raw, labels_test,
            decimation_factor=params.DECIMATION_FACTOR_PREPROC,
            tdfilt_numer=numer, tdfilt_denom=denom,
            #reref_channel_id=params.REREF_CHANNEL_ID,
            #power=True,
            #mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
            scaler=scaler)
    #labels_test = labels_test
    
    # Epoch the test data
    print 'X_test_preproc.shape:', X_test_preproc.shape
    X_epoch_list_test_rh = utils.create_epochs(
            X_test_preproc, labels_test[:, 0], params.EPOCH_OFFSET_SAMPLES)
    X_epoch_list_test_lh = utils.create_epochs(
            X_test_preproc, labels_test[:, 1], params.EPOCH_OFFSET_SAMPLES)
    X_epoch_list_test_all = []
    X_epoch_list_test_all.extend(X_epoch_list_test_rh)
    X_epoch_list_test_all.extend(X_epoch_list_test_lh)
    label_list_test_all = []
    label_list_test_all.extend([[1.0, 0.0]] * len(X_epoch_list_test_rh))
    label_list_test_all.extend([[0.0, 1.0]] * len(X_epoch_list_test_lh))

    # Dummy set for testing
    #X_test_preproc = X_train = np.tile(np.reshape(labels_test[:, 0], (labels_test.shape[0], 1)), [1, params.NUM_CHANNELS])

    # Test the net
    batch_iter_test_valid = EpochBatchIterator(X_epoch_list_test_all, batch_size=params.BATCH_SIZE)
    nnet.batch_iterator_train = None
    nnet.batch_iterator_test = batch_iter_test_valid
    indices_test = np.arange(len(X_epoch_list_test_all))
    print TAG, 'Testing the net...'
    predictions = nnet.predict_proba(indices_test)
    print TAG, 'predictions size:', predictions.shape
    labels_names = ['rh', 'lh', 'idle']

    # Find the thresholds
    tpr_targets = (0.4, 0.4, 0.0)
    #p_thresholds = utils.calculate_auroc(labels_test, predictions, labels_names, tpr_targets, plot=True)
    print 'len(label_list_test_all):', len(label_list_test_all)
    labels_test_arr = np.zeros((len(label_list_test_all), len(label_list_test_all[0])), np.float32)
    for i_label in range(len(label_list_test_all)):
        labels_test_arr[i_label, :] = label_list_test_all[i_label]
    print 'labels_test_arr.shape:', labels_test_arr.shape
    utils.calculate_auroc(labels_test_arr, predictions, labels_names, tpr_targets, plot=True)
    p_thresholds = (0.7, 0.7, 0.7)
    print 'p_thresholds:', p_thresholds

    # Create the control signal time series
    # p_threshold_rh = 0.754
    # p_threshold_lh = 0.379
    print 'predictions.dtype:', predictions.dtype
    predictions_rh = np.copy(predictions[:, 0])
    predictions_lh = np.copy(predictions[:, 1])
    predictions_rh_bin = np.copy(predictions_rh)
    predictions_lh_bin = np.copy(predictions_lh)
    predictions_rh_bin[predictions_rh_bin < p_thresholds[0]] = 0.0
    predictions_rh_bin[predictions_rh_bin >= p_thresholds[0]] = 1.0
    predictions_lh_bin[predictions_lh_bin < p_thresholds[1]] = 0.0
    predictions_lh_bin[predictions_lh_bin >= p_thresholds[1]] = 1.0
    predictions_res_bin = predictions_rh_bin - predictions_lh_bin
    labels_test_res_arr = labels_test_arr[:, 0] - labels_test_arr[:, 1]

    # Create the cursor position time series
    cursor_pos_series = np.zeros(predictions_res_bin.shape)
    for i_time in range(1, predictions_res_bin.shape[0]):
        cursor_pos_series[i_time] = cursor_pos_series[i_time-1] + predictions_res_bin[i_time]

    # Plot prediction confidence histograms
    print 'predictions_rh:\n', predictions_rh
    print 'predictions_lh:\n', predictions_lh
    print 'predictions_rh_bin:\n', predictions_rh_bin
    print 'predictions_lh_bin:\n', predictions_lh_bin
    print 'labels_test_res_arr:\n', labels_test_res_arr
    plt.hist(x=predictions_rh, bins=100, range=(0.0, 1.0), color='r', alpha=0.5, label='rh')
    plt.hist(x=predictions_lh, bins=100, range=(0.0, 1.0), color='b', alpha=0.5, label='lh')
    #plt.title("Right hand")
    plt.show()

    # Plot the timeline
    #print 'indices_test.shape:', indices_test.shape
    #print 'labels_test_res.shape:', labels_test_res_arr.shape
    plt.plot(indices_test, labels_test_res_arr, 'b-', label='event labels')
    plt.plot(indices_test, predictions_res_bin, 'rx', label='predictions')
    plt.plot(indices_test, 0.001*cursor_pos_series, 'r-', label='cursor x pos')
    #plt.plot(indices_test, labels_test[:, 0])
    #plt.plot(indices_test, predictions[:, 0])
    plt.legend(loc='lower right')
    plt.show()

    print 'Main terminates.'

