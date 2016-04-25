

import params
import nolearn
import lasagne
import theano
import logging


TAG = '[nnfactory]'


########################################################################################################################

def nn_loss_func(x, t):
    #logging.debug('%s Loss: x shape: %s', TAG, type(x))
    #logging.debug('%s Loss: t type: %s', TAG, type(t))
    loss_value = lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(x, t))
    #logging.debug('%s loss_value: %f', TAG, loss_value)
    #print TAG, 'loss_value:', loss_value

    return loss_value


########################################################################################################################

def create_nn(
        nn_type,
        nn_input_shape,
        nn_output_shape,
        num_max_training_epochs):

    if 'gtec' in nn_type:
        if 'CovMat' in nn_type:
            if 'small' in nn_type:
                nnet, layer_obj = create_nn_gtec_CovMat_small(nn_input_shape, nn_output_shape, num_max_training_epochs)
            elif 'medium' in nn_type:
                nnet, layer_obj = create_nn_gtec_CovMat_medium(nn_input_shape, nn_output_shape, num_max_training_epochs)
        elif 'Img' in nn_type:
            if 'small' in nn_type:
                nnet, layer_obj = create_nn_gtec_Img_small(nn_input_shape, nn_output_shape, num_max_training_epochs)
            elif 'medium' in nn_type:
                nnet, layer_obj = create_nn_gtec_Img_medium(nn_input_shape, nn_output_shape, num_max_training_epochs)
        elif 'Seq' in nn_type:
            if 'recurrent' in nn_type:
                nnet, layer_obj = create_nn_gtec_Seq_recurrent(nn_input_shape, nn_output_shape, num_max_training_epochs)
            elif 'lstm' in nn_type:
                nnet, layer_obj = create_nn_gtec_Seq_lstm(nn_input_shape, nn_output_shape, num_max_training_epochs)
        elif 'RC' in nn_type:
            nnet, layer_obj = create_nn_gtec_RC(nn_input_shape, nn_output_shape, num_max_training_epochs)
        elif 'LC' in nn_type:
            nnet, layer_obj = create_nn_gtec_LC(nn_input_shape, nn_output_shape, num_max_training_epochs)
        else:
            logging.critical('%s Unknown NN type: %s', TAG, nn_type)
    elif 'biosemi' in nn_type:
        if 'CovMat' in nn_type:
            if 'small' in nn_type:
                nnet, layer_obj = create_nn_biosemi_CovMat_small(nn_input_shape, nn_output_shape, num_max_training_epochs)
            elif 'medium' in nn_type:
                    nnet, layer_obj = create_nn_biosemi_CovMat_medium(nn_input_shape, nn_output_shape, num_max_training_epochs)
        elif 'Img' in nn_type:
            if 'small' in nn_type:
                nnet, layer_obj = create_nn_biosemi_Img_small(nn_input_shape, nn_output_shape, num_max_training_epochs)
            elif 'medium' in nn_type:
                nnet, layer_obj = create_nn_biosemi_Img_medium(nn_input_shape, nn_output_shape, num_max_training_epochs)
        elif 'Seq' in nn_type:
            if 'recurrent' in nn_type:
                nnet, layer_obj = create_nn_biosemi_Seq_recurrent(nn_input_shape, nn_output_shape, num_max_training_epochs)
            elif 'lstm' in nn_type:
                nnet, layer_obj = create_nn_biosemi_Seq_lstm(nn_input_shape, nn_output_shape, num_max_training_epochs)
        else:
            logging.critical('%s Unknown NN type: %s', TAG, nn_type)
    elif 'gal' in nn_type:
        if 'TxC' in nn_type:
            if 'small' in nn_type:
                nnet, layer_obj = create_nn_gal_TxC_small(nn_input_shape, nn_output_shape, num_max_training_epochs)
            else:
                logging.critical('%s Unknown NN type: %s', TAG, nn_type)
        if 'Seq' in nn_type:
            if 'recurrent' in nn_type:
                nnet, layer_obj = create_nn_gal_Seq_recurrent(nn_input_shape, nn_output_shape, num_max_training_epochs)
            elif 'lstm' in nn_type:
                nnet, layer_obj = create_nn_gal_Seq_lstm(nn_input_shape, nn_output_shape, num_max_training_epochs)
        else:
            logging.critical('%s Unknown NN type: %s', TAG, nn_type)
    else:
        logging.critical('%s Unknown NN type.', TAG)

    return nnet, layer_obj


########################################################################################################################

def create_nn_biosemi_Img_medium(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=(None, nn_input_shape[0], nn_input_shape[1], nn_input_shape[2]), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(3, 3), name='Conv_1')
    #layer_obj = lasagne.layers.LocalResponseNormalization2DLayer(layer_obj, name='LRN_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(5, 5), name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(3, 3), name='Conv_3')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_3')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=32, filter_size=(3, 3), name='Conv_4')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_4')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
            nonlinearity=theano.tensor.nnet.sigmoid, name='Output')

    nnet = nolearn.lasagne.NeuralNet(layer_obj,
            update=lasagne.updates.adam, update_learning_rate=0.001,    # adam
            #update=lasagne.updates.nesterov_momentum, update_learning_rate=0.01, update_momentum=0.9,  # Nesterov
            objective_loss_function=nn_loss_func,
            regression=True, max_epochs=num_max_training_epochs,
            y_tensor_type=theano.tensor.matrix, verbose=1)

    return nnet, layer_obj

########################################################################################################################

def create_nn_biosemi_TxC_small(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=(None, 1, nn_input_shape[0], nn_input_shape[1]), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(1, 8), stride=(1, 4), name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(1, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_biosemi_TxC_small(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=(None, 1, nn_input_shape[0], nn_input_shape[1]), name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(1, 8), stride=(1, 4), name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(1, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_biosemi_CovMat_medium(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8,
            filter_size=(17, 17), name='Conv_1')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8,
            filter_size=(17, 17), name='Conv_2')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(5, 5), name='Conv_3')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_3')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(5, 5), name='Conv_4')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_4')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(5, 5), name='Conv_5')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_5')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_biosemi_CovMat_small(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(
            shape=nn_input_shape, name='Input')
    #logging.debug('%s create_nn_biosemi_CovMat_small input shape: %s', TAG, str(nn_input_shape))
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8,
            filter_size=(3, 3), name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8,
            filter_size=(3, 3), name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_2')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_gtec_CovMat_small(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(5, 5), name='Conv_1')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(5, 5), name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_2')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_gtec_CovMat_medium(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(5, 5), name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(5, 5), name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(3, 3), name='Conv_3')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_3')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(3, 3), name='Conv_4')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_4')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_biosemi_Seq_recurrent(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    #
    layer_obj = lasagne.layers.RecurrentLayer(layer_obj, num_units=128, name='Recurr_1')
    #
    layer_obj = lasagne.layers.RecurrentLayer(layer_obj, num_units=128, name='Recurr_2')
    #
    #layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    #layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    #layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    #layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

# Recuttent Layer in shape: (batch_size, n_time_steps, n_features_1, n_features_2, ...)
def create_nn_gtec_Seq_recurrent(nn_input_shape, nn_output_shape, num_max_training_epochs):
    #logging.debug('%s create_nn_gtec_Seq_recurrent input shape: %s', TAG, str(nn_input_shape))
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    #
    layer_obj = lasagne.layers.RecurrentLayer(layer_obj, num_units=128, name='Recurr_1')
    #
    #layer_obj = lasagne.layers.RecurrentLayer(layer_obj, num_units=64, name='Recurr_2')
    #
    layer_obj = lasagne.layers.SliceLayer(layer_obj, indices=-1, axis=1)    # axis 0 is the batch axis
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

# Recuttent Layer in shape: (batch_size, n_time_steps, n_features_1, n_features_2, ...)
def create_nn_gtec_RC(nn_input_shape, nn_output_shape, num_max_training_epochs):
    #logging.debug('%s create_nn_gtec_RC input shape: %s', TAG, str(nn_input_shape))
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    #
    filter_size_Conv_1 = (32, 1)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=filter_size_Conv_1, name='Conv_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=1, filter_size=(1, 1), name='Conv_2')
    #
    layer_obj = lasagne.layers.ReshapeLayer(layer_obj,
            shape=([0],
                    (nn_input_shape[2] - (filter_size_Conv_1[0] - 1)),
                    nn_input_shape[3]),
            name='Reshape_2')
    #
    layer_obj = lasagne.layers.RecurrentLayer(layer_obj, num_units=128, name='Recurr_1')
    #
    #layer_obj = lasagne.layers.RecurrentLayer(layer_obj, num_units=64, name='Recurr_2')
    #
    layer_obj = lasagne.layers.SliceLayer(layer_obj, indices=-1, axis=1)    # axis 0 is the batch axis
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_gtec_LC(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    #
    filter_size_Conv_1 = (17, 1)
    stride_Conv_1 = (2, 1)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8,
            filter_size=filter_size_Conv_1, stride=stride_Conv_1, name='Conv_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=1,
            filter_size=(1, 1), name='Conv_2')
    #
    layer_obj = lasagne.layers.ReshapeLayer(layer_obj,
            shape=([0],
                    (nn_input_shape[2] - (filter_size_Conv_1[0] - 1)) / stride_Conv_1[0],
                    nn_input_shape[3]),
            name='Reshape_2')
    #
    layer_obj = lasagne.layers.LSTMLayer(layer_obj, num_units=128, name='LSTM_1')
    #
    #layer_obj = lasagne.layers.LSTMLayer(layer_obj, num_units=128, name='LSTM_2')
    #
    layer_obj = lasagne.layers.SliceLayer(layer_obj, indices=-1, axis=1)    # axis 0 is the batch axis
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_gtec_Seq_lstm(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    #
    layer_obj = lasagne.layers.LSTMLayer(layer_obj, num_units=128, name='LSTM_1')
    #
    #layer_obj = lasagne.layers.LSTMLayer(layer_obj, num_units=128, name='LSTM_2')
    #
    layer_obj = lasagne.layers.SliceLayer(layer_obj, indices=-1, axis=1)    # axis 0 is the batch axis
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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
def create_nn_biosemi_TxC_medium(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shapenn_input_shape, name='Input')
    # To Conv1DLayer: 3D tensor, with shape (batch_size, num_input_channels, input_length)
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=4, filter_size=(1, 2), stride=(1, 2), name='Conv_1')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(1, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(9, 1), stride=(2, 1), name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_2')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_2')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=8, filter_size=(9, 1), stride=(2, 1),
            name='Conv_3')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_3')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_3')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(3, 3),
            name='Conv_4')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_4')
    #
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(3, 3),
            name='Conv_5')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_5')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_5')
    #
    #layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(9, 5),
    #        name='Conv_4')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_4')
    #layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_4')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512,
            name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512,
            name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_gal_TxC_small(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(1, 5), name='Conv_1')
    layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 2), name='MaxPool_1')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_1')
    #
    #layer_obj = lasagne.layers.Conv2DLayer(layer_obj, num_filters=16, filter_size=(5, 5), name='Conv_2')
    #layer_obj = lasagne.layers.MaxPool2DLayer(layer_obj, pool_size=(2, 1), name='MaxPool_2')
    #layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_2')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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

def create_nn_gal_Seq_lstm(nn_input_shape, nn_output_shape, num_max_training_epochs):
    layer_obj = lasagne.layers.InputLayer(shape=nn_input_shape, name='Input')
    #
    layer_obj = lasagne.layers.LSTMLayer(layer_obj, num_units=128, name='LSTM_1')
    #
    #layer_obj = lasagne.layers.LSTMLayer(layer_obj, num_units=128, name='LSTM_2')
    #
    layer_obj = lasagne.layers.SliceLayer(layer_obj, indices=-1, axis=1)    # axis 0 is the batch axis
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_98')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_98')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=512, name='Dense_99')
    layer_obj = lasagne.layers.DropoutLayer(layer_obj, p=0.5, name='Dropout_99')
    #
    layer_obj = lasagne.layers.DenseLayer(layer_obj, num_units=nn_output_shape,
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
