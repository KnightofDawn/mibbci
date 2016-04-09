

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


TAG = '[nnutils] '




########################################################################################################################

def nn_loss_func(x, t):
    return lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(x, t))


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

# LeNet: 46x46 -> 20x42x42 -> 20x14x14 -> 100x10x10 -> 100x5x5 -> 200x1 (dense) -> 6x1 (multinom logreg)
# HEDJ
def create_nn_big(num_max_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=(None, params.NUM_CHANNELS, params.WINDOW_SIZE_DEC_SAMPLES), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=8, filter_size=3,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_1')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=16, filter_size=3,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_2')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=32, filter_size=17,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_3')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=32, filter_size=3,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_4')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=32, filter_size=3,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_5')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_5')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_5')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=params.NUM_DENSE_UNITS,
            nonlinearity=params.INTERNAL_NONLINEARITY_DENSE, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=params.NUM_DENSE_UNITS,
            nonlinearity=params.INTERNAL_NONLINEARITY_DENSE, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=params.NUM_EVENT_TYPES,
            nonlinearity=theano.tensor.nnet.sigmoid, name='Output')

    nnet = nolearn.lasagne.NeuralNet(layer_obj,
            #update=lasagne.updates.adam,  # adam, adagrad, nesterov_momentum,
            #update_learning_rate=0.001,    # adam
            update=lasagne.updates.nesterov_momentum,   # Nesterov
            update_learning_rate=0.01,  # Nesterov
            update_momentum=0.9,  # Nesterov
            objective_loss_function=nn_loss_func,
            regression=True, max_epochs=num_max_epochs,
            y_tensor_type=theano.tensor.matrix, verbose=1)
    #nnet.initialize()
    #print 'nnet.predict_iter_: ', nnet.predict_iter_

    return nnet, layer_obj




########################################################################################################################

# LeNet: 46x46 -> 20x42x42 -> 20x14x14 -> 100x10x10 -> 100x5x5 -> 200x1 (dense) -> 6x1 (multinom logreg)
# HEDJ
def create_nn_medium(num_inputs, num_outputs, num_max_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=(None, num_inputs[0], num_inputs[1]), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=16, filter_size=3,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_1')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=16, filter_size=3,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_2')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=32, filter_size=2,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_3')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_3')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=params.NUM_DENSE_UNITS,
            nonlinearity=params.INTERNAL_NONLINEARITY_DENSE, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=params.NUM_DENSE_UNITS,
            nonlinearity=params.INTERNAL_NONLINEARITY_DENSE, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=num_outputs,
            nonlinearity=theano.tensor.nnet.sigmoid, name='Output')

    nnet = nolearn.lasagne.NeuralNet(layer_obj,
            # update=lasagne.updates.adam,  # adam, adagrad, nesterov_momentum,
            # update_learning_rate=0.001,    # adam
            update=lasagne.updates.nesterov_momentum,  # Nesterov
            update_learning_rate=0.01,  # Nesterov
            update_momentum=0.9,  # Nesterov
            objective_loss_function=nn_loss_func,
            regression=True, max_epochs=num_max_epochs,
            y_tensor_type=theano.tensor.matrix, verbose=1)
    #nnet.initialize()
    #print 'nnet.predict_iter_: ', nnet.predict_iter_

    return nnet, layer_obj


########################################################################################################################

def train_nn_from_timeseries_data(
        nnet,
        X_train, labels_train,
        num_train_data_instances,
        validation_data_ratio=0.2):

    # Dummy set for testing
    #X_train = np.tile(np.reshape(labels_train[:, 0], (labels_train.shape[0], 1)), [1, params.NUM_CHANNELS])

    # Create the batch iterators
    print 'X_train.shape:', X_train.shape
    index_start_validation = int((1.0 - validation_data_ratio) * X_train.shape[0])
    print(TAG, 'index_start_validation: ', index_start_validation)
    batch_iter_train_base = TimeSeriesBatchIterator(X_train[:index_start_validation],
        labels_train[:index_start_validation], batch_size=params.BATCH_SIZE)
    batch_iter_train_valid = TimeSeriesBatchIterator(X_train[index_start_validation:],
        labels_train[index_start_validation:], batch_size=params.BATCH_SIZE)

    # Add the batch iterators to the net
    nnet.batch_iterator_train = batch_iter_train_base
    nnet.batch_iterator_test = batch_iter_train_valid

    train_indices = np.zeros(num_train_data_instances, dtype=int) - 1    # Passing full zeros, the batch iterator will pull data instances randomly
    print(TAG, 'Fitting the classifier...')
    #t_start_fit = time.time()
    nnet.fit(train_indices, train_indices)


########################################################################################################################

def train_nn_from_epoch_data(nnet,
        X_epoch_list, label_list,
        num_train_data_instances,
        validation_data_ratio=0.2):

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

def train_nn_timeseries(
        nnet,
        X_train, labels_train,
        num_train_data_instances,
        plot_history=False):

    # Init and train the NN
    validation_data_ratio = params.VALIDATION_DATA_RATIO
    train_nn_from_timeseries_data(
            nnet, X_train, labels_train, num_train_data_instances, validation_data_ratio)
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

def train_nn_epochs(
        nnet,
        X_epoch_list,
        label_list,
        num_train_data_instances,
        plot_history=False):

    # Init and train the NN
    validation_data_ratio = params.VALIDATION_DATA_RATIO
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

