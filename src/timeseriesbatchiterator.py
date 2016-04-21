

import params
import numpy as np
from nolearn.lasagne import BatchIterator
import logging

TAG = '[CustomBatchIterator] '



class TimeSeriesBatchIterator(BatchIterator):

    def __init__(self, data, labels=None,
            window_size_samples=-1, num_outputs=-1,     # TODO nicer
            *args, **kwargs):

        super(TimeSeriesBatchIterator, self).__init__(*args, **kwargs)

        self._window_size_samples = window_size_samples
        logging.debug('%s self._window_size_samples: %d', TAG, self._window_size_samples)

        self._X = data
        logging.debug('%s self._X size: %d, %d, %d', TAG, self._X.shape[0], self._X.shape[1], self._X.shape[2])

        self._num_conv_dims = 2     # be 2 the default; num_conv_dims

        if labels is not None:
            self._y = labels
            logging.debug('%s self._y size: %d, %d', TAG, self._y.shape[0], self._y.shape[1])
            logging.debug('%s np.sum(self._y): %f', TAG, np.sum(self._y))

        if self._num_conv_dims == 2:    # Conv2D case
            self._X_padded = np.concatenate(
                    (np.zeros((self._window_size_samples-1, self._X.shape[1], self._X.shape[2])), self._X),
                    axis=0)     # Padding with zeros
            self._X_buf = np.zeros(
                    (self.batch_size, self._window_size_samples, self._X.shape[1], self._X.shape[2]),
                    np.float32)   # Pre-allocating buffers
        else:
            logging.error('%s Wrong num_conv_dims: %d', TAG, num_conv_dims);

        print TAG, 'self._X_padded size:', self._X_padded.shape
        print TAG, 'self._X_buf size:', self._X_buf.shape

        self._y_buf = np.zeros([self.batch_size, num_outputs], np.float32)

        print TAG, 'self._X size:', self._X.shape
        print TAG, 'self._X_padded size:', self._X_padded.shape

        self._counter = 0

    # End of init

    @staticmethod
    def create_X_instance(X_window, num_conv_dims):
        if num_conv_dims == 1:
            #X_instance = X_window[::-1][::params.DECIMATION_FACTOR_NN].transpose()  # Reversing to get the latest point after the downsampling
            X_instance = X_window.transpose()
        elif num_conv_dims == 2:
            #X_instance = X_window.transpose((0, 2, 1))
            X_instance = X_window
        return X_instance

    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(TimeSeriesBatchIterator, self).transform(X_indices, y_indices)
        #print(TAG, "X_indices part:\n", X_indices[20:30])
        #[count] = X_indices.shape
        #X_batch = self._X_buf[:count]
        #y_batch = self._Y_buf[:count]
        num_indices = X_indices.shape[0]

        X_batch = self._X_buf[:num_indices]
        y_batch = self._y_buf[:num_indices]

        if self._num_conv_dims == 2:     # Conv2D case
            for i_index, i_time in enumerate(X_indices):

                if i_time < 0:
                    i_time = np.random.randint((self._window_size_samples-1), self._X.shape[0])

                X_window = self._X_padded[i_time:i_time+self._window_size_samples, :]
                #print(TAG, "X_window part:\n", X_window[0, 2:6, 20:25])
                #print("X_window size:", X_window.shape)
                X_batch[i_index] = TimeSeriesBatchIterator.create_X_instance(
                        X_window, num_conv_dims=self._num_conv_dims)
                #X_batch[i_index] = X_window[::-1][:, ::params.DEC_FACTOR].transpose((0, 2, 1))      # Reversing to get the latest point after the downsampling
                #print TAG, "X_batch[i_index]:\n", X_batch[i_index]
                if y_indices is not None:
                    y_batch[i_index] = self._y[i_time]
                    #print TAG, "y_batch[i_index]:\n", y_batch[i_index]
                    #if y_batch[i_index] > 0.0:
                    #    logging.debug('y_batch[i_index] > 0.0: %f', y_batch[i_index])
                    #else:
                    #    logging.debug('y_batch[i_index]: %f', y_batch[i_index])
                else:
                    self._counter += 1
                    if self._counter % 5000 == 0:
                        logging.debug('%s self._counter: %d', TAG, self._counter)

                #logging.debug('%s np.sum(y_batch): %f', TAG, np.sum(y_batch))
                #logging.debug('%s np.sum(self._y): %f', TAG, np.sum(self._y))

        else:
            print TAG, 'Wrong num_conv_dims:', num_conv_dims

        if y_indices is None:
            y_batch = None

        return X_batch, y_batch
