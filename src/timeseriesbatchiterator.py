

import params
import numpy as np
from nolearn.lasagne import BatchIterator
import logging

TAG = '[timeseriesbatchiterator]'



class TimeSeriesBatchIterator(BatchIterator):

    def __init__(self,
            data, labels=None,
            nn_type=None,
            window_size_samples=-1, num_nn_outputs=-1,     # TODO nicer
            *args, **kwargs):

        logging.debug('%s Initializing TimeSeriesBatchIterator...', TAG)

        super(TimeSeriesBatchIterator, self).__init__(*args, **kwargs)

        self._nn_type = nn_type

        self._window_size_samples = window_size_samples
        logging.debug('%s self._window_size_samples: %d', TAG, self._window_size_samples)

        self._X = data
        logging.debug('%s self._X size: %s', TAG, str(self._X.shape))

        self._num_conv_dims = 2     # be 2 the default; num_conv_dims

        if labels is not None:
            self._y = labels
            logging.debug('%s self._y size: %s', TAG, str(self._y.shape))
            logging.debug('%s np.sum(self._y, axis=0): %s', TAG, str(np.sum(self._y, axis=0).tolist()))

        if self._num_conv_dims == 2:    # Conv2D case
            if 'CovMat' in self._nn_type:
                self._X_padded = np.concatenate(
                        (np.zeros((self._window_size_samples-1, self._X.shape[1])), self._X),
                        axis=0)     # Padding with zeros
                self._X_buf = np.zeros(
                        (self.batch_size, 1, self._X.shape[1], self._X.shape[1]),
                        np.float32)   # Pre-allocating buffers
            elif 'Img' in self._nn_type:
                self._X_padded = np.concatenate(
                        (np.zeros((self._window_size_samples-1, self._X.shape[1], self._X.shape[2])), self._X),
                        axis=0)     # Padding with zeros
                self._X_buf = np.zeros(
                        (self.batch_size, self._window_size_samples, self._X.shape[1], self._X.shape[2]),
                        np.float32)   # Pre-allocating buffers
            elif 'TxC' in self._nn_type:
                self._X_padded = np.concatenate(
                        (np.zeros((self._window_size_samples-1, self._X.shape[1])), self._X),
                        axis=0)     # Padding with zeros
                self._X_buf = np.zeros(
                        (self.batch_size, 1, self._window_size_samples, self._X.shape[1]),
                        np.float32)   # Pre-allocating buffers
            elif 'Seq' in self._nn_type:
                self._X_padded = np.concatenate(
                        (np.zeros((self._window_size_samples-1, self._X.shape[1])), self._X),
                        axis=0)     # Padding with zeros
                self._X_buf = np.zeros(
                        (self.batch_size, self._window_size_samples, self._X.shape[1]),
                        np.float32)   # Pre-allocating buffers
        else:
            logging.error('%s Wrong num_conv_dims: %d', TAG, num_conv_dims);

        logging.debug('%s self._X_padded size: %s', TAG, str(self._X_padded.shape))
        logging.debug('%s self._X_buf size: %s', TAG, str(self._X_buf.shape))

        self._y_buf = np.zeros([self.batch_size, num_nn_outputs], np.float32)

        self._counter = 0

        logging.debug('%s TimeSeriesBatchIterator initiaized.', TAG)

    # End of init

    @staticmethod
    def create_X_instance(X_window, num_conv_dims, nn_type):
        if num_conv_dims == 1:
            pass
        elif num_conv_dims == 2:
            if 'CovMat' in nn_type:
                X_instance = np.dot(X_window.T, X_window) / float(X_window.shape[0] - 1.0)
            elif 'TxC' in nn_type:
                X_instance = X_window
            elif 'Seq' in nn_type:
                X_instance = X_window
            else:
                logging.critical('%s Unknown NN type.', TAG)

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
                        X_window, num_conv_dims=self._num_conv_dims, nn_type=self._nn_type)
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
