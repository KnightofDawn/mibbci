

from timeseriesbatchiterator import TimeSeriesBatchIterator
from epochbatchiterator import EpochBatchIterator
import utils
import params
import nolearn
import lasagne
import theano
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cPickle
import logging


TAG = '[nnutils]'




########################################################################################################################

def load_nn(create_nn_func, filename):

    #nnet, nnet_last_layer = create_nn()

    # Load the NN with load_params_from
    nnet, _ = create_nn_func()
    nnet.load_params_from(filename)


    # Load the NN with set_all_param_values
    # arr_0, arr_1, ... by default
    #with np.load(param_vals_filename) as f:
    #    nnet_layer_param_values_loaded = [f['arr_%d' % i] for i in range(len(f.files))]
    #nnet_layer_param_values_loaded = np.load(filename)
    #print 'nnet_layer_param_values_loaded[0]:\n', nnet_layer_param_values_loaded[0]
    #print 'nnet_layer_param_values_loaded:\n', nnet_layer_param_values_loaded
    #print 'type(nnet_layer_param_values_loaded):', type(nnet_layer_param_values_loaded)
    #print 'type(nnet_layer_param_values_loaded):', type(nnet_layer_param_values_loaded[0])
    #print 'len(nnet_layer_param_values_loaded[0]):', len(nnet_layer_param_values_loaded[0])
    #lasagne.layers.set_all_param_values(nnet_last_layer, nnet_layer_param_values_loaded[0])
    #lasagne.layers.set_all_param_values(nnet_last_layer, nnet_layer_param_values_saved)
    #nnet_param_vals_loaded =
    #nnet_params = lasagne.layers.get_all_params(nnet_last_layer)
    #for param, val in zip(nnet_params, nnet_param_vals_loaded):
    #    param.set_value(val)

    print TAG, ' nn loaded from ', filename

    return nnet




########################################################################################################################


def save_nn(nnet, filename):

    nnet.save_params_to(filename)

    # nnet_layer_params = lasagne.layers.get_all_params(nnet_last_layer)
    # print 'type(nnet_layer_params):', type(nnet_layer_params)
    # nnet_layer_param_vals_list = [param.get_value() for param in nnet_layer_params]
    # print 'type(nnet_layer_param_vals):', type(nnet_layer_param_vals_list)
    # print 'type(nnet_layer_param_vals[0]):', type(nnet_layer_param_vals_list[0])
    # print 'len(nnet_layer_param_vals):', len(nnet_layer_param_vals_list)
    # nnet_layer_param_vals_arr = np.asarray(nnet_layer_param_vals_list)
    # print 'nnet_layer_param_vals_arr.shape:', nnet_layer_param_vals_arr.shape
    #nnet_layer_param_values_saved = lasagne.layers.get_all_param_values(nnet_last_layer)
    #print 'len(nnet_layer_param_values_saved):', len(nnet_layer_param_values_saved)
    # print 'nnet_layer_param_values_saved:\n', nnet_layer_param_values_saved
    # np.savez(filename, *nnet_layer_param_values_saved)
    #np.savez(filename, nnet_layer_param_values_saved)

    print TAG, ' nn saved to', filename


########################################################################################################################

def train_nn_from_timeseries_data(
        nnet,
        X_train, labels_train,
        window_size_samples,
        num_outputs,
        num_train_data_instances,
        validation_data_ratio=0.4):

    # Dummy set for testing
    #X_train = np.tile(np.reshape(labels_train[:, 0], (labels_train.shape[0], 1)), [1, params.NUM_CHANNELS])

    # Create the batch iterators
    print 'X_train.shape:', X_train.shape
    index_start_validation = int((1.0 - validation_data_ratio) * X_train.shape[0])
    logging.debug('%s index_start_validation: %d', TAG, index_start_validation)
    batch_iter_train_base = TimeSeriesBatchIterator(
            data=X_train[:index_start_validation],
            labels=labels_train[:index_start_validation],
            window_size_samples=window_size_samples,
            num_outputs=num_outputs,
            batch_size=params.BATCH_SIZE)
    batch_iter_train_valid = TimeSeriesBatchIterator(
            X_train[index_start_validation:],
            labels=labels_train[index_start_validation:],
            window_size_samples=window_size_samples,
            num_outputs=num_outputs,
            batch_size=params.BATCH_SIZE)

    # Add the batch iterators to the net
    nnet.batch_iterator_train = batch_iter_train_base
    nnet.batch_iterator_test = batch_iter_train_valid

    train_indices = np.zeros(num_train_data_instances, dtype=int) - 1    # Passing full zeros, the batch iterator will pull data instances randomly
    print TAG, 'Fitting the classifier...'
    #t_start_fit = time.time()
    nnet.fit(train_indices, train_indices)


########################################################################################################################

def train_nn_from_epoch_data(nnet,
        X_epoch_list, label_list,
        num_train_data_instances,
        validation_data_ratio=0.4):

    # Dummy set for testing
    #X_train = np.tile(np.reshape(labels_train[:, 0], (labels_train.shape[0], 1)), [1, params.NUM_CHANNELS])

    # Create the list of the class labels
    #label_list = []

    # Create the batch iterators
    index_start_validation = int((1.0 - validation_data_ratio) * len(X_epoch_list))
    print(TAG, 'index_start_validation: ', index_start_validation)
    batch_iter_train_base = EpochBatchIterator(X_epoch_list[:index_start_validation],
        label_list[:index_start_validation], batch_size=params.BATCH_SIZE)
    batch_iter_train_valid = EpochBatchIterator(X_epoch_list[index_start_validation:],
        label_list[index_start_validation:], batch_size=params.BATCH_SIZE)

    # Add the batch iterators to the net
    nnet.batch_iterator_train = batch_iter_train_base
    nnet.batch_iterator_test = batch_iter_train_valid

    train_indices = np.zeros(num_train_data_instances, dtype=int) - 1    # Passing full zeros, the batch iterator will pull data instances randomly
    print(TAG, 'Fitting the classifier...')
    #t_start_fit = time.time()
    nnet.fit(train_indices, train_indices)


########################################################################################################################

def train_nn_from_timeseries(
        nnet,
        X_train, labels_train,
        window_size_samples,
        num_outputs,
        num_train_data_instances,
        validation_data_ratio=0.4,
        plot_history=False):

    # Init and train the NN
    train_nn_from_timeseries_data(
            nnet,
            X_train, labels_train,
            window_size_samples,
            num_outputs,
            num_train_data_instances,
            validation_data_ratio)
    if plot_history:
        train_loss = np.array([i["train_loss"] for i in nnet.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in nnet.train_history_])
        plt.plot(train_loss, linewidth=3, label="train")
        plt.plot(valid_loss, linewidth=3, label="valid")
        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        #plt.ylim(1e-3, 1e-2)
        #plt.yscale("log")
        plt.show()

    return nnet


########################################################################################################################

def train_nn_from_epochs(
        nnet,
        X_epoch_list,
        label_list,
        num_train_data_instances,
        validation_data_ratio=0.4,
        plot_history=False):

    # Init and train the NN
    train_nn_from_epoch_data(
            nnet, X_epoch_list, label_list, num_train_data_instances, validation_data_ratio)
    if plot_history:
        train_loss = np.array([i["train_loss"] for i in nnet.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in nnet.train_history_])
        plt.plot(train_loss, linewidth=3, label="train")
        plt.plot(valid_loss, linewidth=3, label="valid")
        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        #plt.ylim(1e-3, 1e-2)
        #plt.yscale("log")
        plt.show()

    return nnet
