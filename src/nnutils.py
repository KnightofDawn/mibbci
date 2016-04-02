

from custombatchiterator import CustomBatchIterator
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

def fit_scaler(X_raw):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_raw)

    return scaler


########################################################################################################################

def preprocess(X_raw, scaler):

    return scaler.transform(X_raw)


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
def create_nn_big():
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
            regression=True, max_epochs=params.NUM_MAX_TRAIN_EPOCHS,
            y_tensor_type=theano.tensor.matrix, verbose=1)
    #nnet.initialize()
    #print 'nnet.predict_iter_: ', nnet.predict_iter_

    return nnet, layer_obj




########################################################################################################################

# LeNet: 46x46 -> 20x42x42 -> 20x14x14 -> 100x10x10 -> 100x5x5 -> 200x1 (dense) -> 6x1 (multinom logreg)
# HEDJ
def create_nn_medium():
    layer_obj = lasagne.layers.InputLayer(shape=(None, params.NUM_CHANNELS, params.WINDOW_SIZE_DEC_SAMPLES), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=16, filter_size=3,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_1')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=32, filter_size=5,
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv1D_2')
    layer_obj = lasagne.layers.MaxPool1DLayer(layer_obj, pool_size=params.MAXPOOL_SIZE, name='MaxPool1D_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv1DLayer(layer_obj, num_filters=64, filter_size=7,
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
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=params.NUM_EVENT_TYPES,
            nonlinearity=theano.tensor.nnet.sigmoid, name='Output')

    nnet = nolearn.lasagne.NeuralNet(layer_obj,
            # update=lasagne.updates.adam,  # adam, adagrad, nesterov_momentum,
            # update_learning_rate=0.001,    # adam
            update=lasagne.updates.nesterov_momentum,  # Nesterov
            update_learning_rate=0.01,  # Nesterov
            update_momentum=0.9,  # Nesterov
            regression=True, max_epochs=params.NUM_MAX_TRAIN_EPOCHS,
            y_tensor_type=theano.tensor.matrix, verbose=1)
    #nnet.initialize()
    #print 'nnet.predict_iter_: ', nnet.predict_iter_

    return nnet, layer_obj


########################################################################################################################

def train_nn_from_data(nnet, X_train, labels_train, validation_data_ratio=0.2):

    # Create the batch iterators
    print 'X_train.shape:', X_train.shape
    index_start_validation = int((1.0 - validation_data_ratio) * X_train.shape[0])
    print(TAG, 'index_start_validation: ', index_start_validation)
    batch_iter_train_base = CustomBatchIterator(X_train[:index_start_validation],
        labels_train[:index_start_validation], batch_size=params.BATCH_SIZE)
    batch_iter_train_valid = CustomBatchIterator(X_train[index_start_validation:],
        labels_train[index_start_validation:], batch_size=params.BATCH_SIZE)

    # Add the batch iterators to the net
    nnet.batch_iterator_train = batch_iter_train_base
    nnet.batch_iterator_test = batch_iter_train_valid

    train_indices = np.zeros(params.NUM_TRAIN_DATA_INSTANCES, dtype=int) - 1    # Passing full zeros, the batch iterator will pull data instances randomly
    print(TAG, 'Fitting the classifier...')
    #t_start_fit = time.time()
    nnet.fit(train_indices, train_indices)




########################################################################################################################


def train_nn(nnet, data_filename_list, scaler=None, plot_history=False):

    # Load the training data
    X_train_raw, labels_train = utils.load_data(data_filename_list)

    # Preprocess the raw data
    # trunc_x_from = 2000
    # trunc_x_to = 10000
    # X_preproc = preprocess(X_train_raw[trunc_x_from:trunc_x_to, :])
    # labels_train = labels_train[trunc_x_from:trunc_x_to, :]
    if scaler is None:
        scaler = fit_scaler(X_train_raw)
        print 'Fit scaler scaler.mean_, scaler.var_:', scaler.mean_, scaler.var_
    else:
        print 'Provided scaler scaler.mean_, scaler.var_:', scaler.mean_, scaler.var_
    X_train_preproc = preprocess(X_train_raw, scaler)
    labels_train = labels_train

    # Init and train the NN
    validation_data_ratio = params.VALIDATION_DATA_RATIO
    train_nn_from_data(nnet, X_train_preproc, labels_train, validation_data_ratio)
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

    # Save the NN
    time_save = datetime.now()
    filename_base = '../models/MIBBCI_NN_{0}{1:02}{2:02}_{3:02}h{4:02}m{5:02}s'.format(
        time_save.year, time_save.month, time_save.day, time_save.hour, time_save.minute, time_save.second)
    filename_nn = filename_base + '.npz'
    save_nn(nnet, filename_nn)

    # Save the preproc stuff
    print 'Before dump scaler.mean_, scaler.var_:', scaler.mean_, scaler.var_
    filename_p = filename_base + '.p'
    cPickle.dump(scaler, open(filename_p, 'wb'))

    # Test-load the NN with load_params_from
    #load_nn(create_nn_medium, filename_nn)

    # Test-load the preproc stuff
    #scaler = cPickle.load(open(filename_p, 'rb'))
    #print 'After load scaler.mean_, scaler.var_:', scaler.mean_, scaler.var_

    return nnet, scaler



