

import params
import nolearn
import lasagne
import theano
import logging


TAG = '[nnfactory]'


########################################################################################################################

def create_nn(nn_type, num_inputs, num_outputs, num_max_training_epochs):

    if nn_type == 'medium_IMG':
        nnet, layer_obj = create_nn_medium_IMG(num_inputs, num_outputs, num_max_training_epochs)
    elif nn_type == 'small_TxC':
        nnet, layer_obj = create_nn_small_TxC(num_inputs, num_outputs, num_max_training_epochs)
    elif nn_type == 'medium_TxC':
        nnet, layer_obj = create_nn_medium_TxC(num_inputs, num_outputs, num_max_training_epochs)
    elif nn_type == 'big':
        nnet, layer_obj = create_nn_big(num_inputs, num_outputs, num_max_training_epochs)

    return nnet, layer_obj


########################################################################################################################

def nn_loss_func(x, t):
    loss_value = lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(x, t))
    #logging.debug('%s loss_value: %f', TAG, loss_value)
    #print TAG, 'loss_value:', loss_value

    return loss_value


########################################################################################################################

def create_nn_medium_IMG(num_inputs, num_outputs, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=(None, num_inputs[0], num_inputs[1], num_inputs[2]), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(3, 3),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(5, 5),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(3, 3),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_3')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_3')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(3, 3),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_4')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_4')
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
            update=lasagne.updates.adam, update_learning_rate=0.001,    # adam
            #update=lasagne.updates.nesterov_momentum, update_learning_rate=0.01, update_momentum=0.9,  # Nesterov
            objective_loss_function=nn_loss_func,
            regression=True, max_epochs=num_max_training_epochs,
            y_tensor_type=theano.tensor.matrix, verbose=1)

    return nnet, layer_obj

########################################################################################################################

def create_nn_small_TxC(num_inputs, num_outputs, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=(None, 1, num_inputs[0], num_inputs[1]), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(1, 8), stride=(1, 4),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(1, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_1')
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
            regression=True, max_epochs=num_max_training_epochs,
            y_tensor_type=theano.tensor.matrix, verbose=1)
    #nnet.initialize()
    #print 'nnet.predict_iter_: ', nnet.predict_iter_

    return nnet, layer_obj

########################################################################################################################

# LeNet: 46x46 -> 20x42x42 -> 20x14x14 -> 100x10x10 -> 100x5x5 -> 200x1 (dense) -> 6x1 (multinom logreg)
# HEDJ
def create_nn_medium_TxC(num_inputs, num_outputs, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=(None, 1, num_inputs[0], num_inputs[1]), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=4, filter_size=(1, 2), stride=(1, 2),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(1, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(9, 1), stride=(2, 1),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(9, 1), stride=(2, 1),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_3')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_3')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(3, 3),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_4')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_4')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(3, 3),
            nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_5')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_5')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_5')
    #
    #layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(9, 5),
    #        nonlinearity=params.INTERNAL_NONLINEARITY_CONV, name='Conv_4')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    #layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=params.DROPOUT_PROBABILITY, name='Dropout_4')
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
            update=lasagne.updates.adam, update_learning_rate=0.001,    # adam
            #update=lasagne.updates.nesterov_momentum, update_learning_rate=0.01, update_momentum=0.9,  # Nesterov
            objective_loss_function=nn_loss_func,
            regression=True, max_epochs=num_max_training_epochs,
            y_tensor_type=theano.tensor.matrix, verbose=1)
    #nnet.initialize()
    #print 'nnet.predict_iter_: ', nnet.predict_iter_

    return nnet, layer_obj


########################################################################################################################

# LeNet: 46x46 -> 20x42x42 -> 20x14x14 -> 100x10x10 -> 100x5x5 -> 200x1 (dense) -> 6x1 (multinom logreg)
# HEDJ
def create_nn_big(num_max_training_epochs):
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
            regression=True, max_epochs=num_max_training_epochs,
            y_tensor_type=theano.tensor.matrix, verbose=1)
    #nnet.initialize()
    #print 'nnet.predict_iter_: ', nnet.predict_iter_

    return nnet, layer_obj
