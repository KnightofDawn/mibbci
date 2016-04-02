'''

TODO
- auto save the roc images
- record more, with 1sec stim time
- write net scheme to the param file somehow
- try 1D AlexNet


http://rosinality.ncity.net/doku.php?id=python:installing_theano

'''

from custombatchiterator import CustomBatchIterator
import params
import nnutils
import utils
import numpy as np
import matplotlib.pyplot as plt
import cPickle


TAG = '[process_nn]'




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    print 'Main started.'

    # Select the data files
    data_filename_train_list = []
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h10m35s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h17m46s.csv')
    #data_filename_train_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h26m54s.csv')
    #data_filename_train_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h26m13s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_18h57m57s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h08m55s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h13m01s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h17m10s_RAW.csv')
    data_filename_train_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h21m49s_RAW.csv')

    # Load/pretrain the net
    #is_net_pretrained = False
    is_net_pretrained = False
    is_net_to_train_more = False
    if is_net_pretrained:
        filename_base = '../models/MIBBCI_NN_20160402_18h29m19s'
        filename_nn = filename_base + '.npz'
        nnet = nnutils.load_nn(nnutils.create_nn_medium, filename_nn)
        filename_p = filename_base + '.p'
        scaler = cPickle.load(open(filename_p, 'rb'))
        print 'Loaded scaler.mean_, scaler.var_:', scaler.mean_, scaler.var_

        if is_net_to_train_more:
            nnet, scaler = nnutils.train_nn(nnet, data_filename_train_list, plot_history=True)
    else:   # If a new net is to be created)
        nnet, _ = nnutils.create_nn_medium()
        nnet, scaler = nnutils.train_nn(nnet, data_filename_train_list, plot_history=True)
        # TODO save net here and not in train

    # Load the test data
    data_filename_test_list = []
    #data_filename_test_list.append('../data/2016-03-26/MIBBCI_REC_20160326_15h32m14s.csv')
    #data_filename_test_list.append('../data/2016-04-02/MIBBCI_REC_20160402_16h37m27s_RAW.csv')
    data_filename_test_list.append('../data/2016-04-02_2/MIBBCI_REC_20160402_19h26m01s_RAW.csv')
    X_test_raw, labels_test = utils.load_data(data_filename_test_list)

    # Pre-process the test data
    X_test_preproc = nnutils.preprocess(X_test_raw, scaler)
    labels_test = labels_test

    # Test the net
    batch_iter_test_valid = CustomBatchIterator(X_test_preproc, batch_size=params.BATCH_SIZE)
    nnet.batch_iterator_train = None
    nnet.batch_iterator_test = batch_iter_test_valid
    indices_test = np.arange(X_test_preproc.shape[0])
    print TAG, 'Testing the net...'
    predictions = nnet.predict_proba(indices_test)
    print TAG, 'predictions size:', predictions.shape
    labels_names = ['rh', 'lh', 'idle']
    tpr_targets = (0.4, 0.4, 0.0)
    p_thresholds = utils.calculate_auroc(labels_test, predictions, labels_names, tpr_targets, plot=True)
    print 'p_thresholds:', p_thresholds

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

    # Plot the timeline
    plt.plot(indices_test, labels_test_res, 'b-', label='event labels')
    plt.plot(indices_test, predictions_res_bin, 'rx', label='predictions')
    plt.plot(indices_test, 0.001*cursor_pos_series, 'r-', label='cursor x pos')
    #plt.plot(indices_test, labels_test[:, 0])
    #plt.plot(indices_test, predictions[:, 0])
    plt.legend(loc='lower right')
    plt.show()

    print 'Main terminates.'

