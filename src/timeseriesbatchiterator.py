

import params
import numpy as np
from nolearn.lasagne import BatchIterator

TAG = '[CustomBatchIterator] '



class TimeSeriesBatchIterator(BatchIterator):

    def __init__(self, data, labels=None,
            window_size_samples=-1, num_outputs=-1,     # TODO nicer
            *args, **kwargs):
        
        super(TimeSeriesBatchIterator, self).__init__(*args, **kwargs)

        self._window_size_samples = window_size_samples
        print 'self._window_size_samples:', self._window_size_samples
        
        self._X = data
        print TAG, 'self._X size:', self._X.shape
        
        self._num_conv_dims = 2     # be 2 the default; num_conv_dims

        if labels is not None:
            self._y = labels
            print TAG, 'self._y size:', self._y.shape

        if self._num_conv_dims == 2:    # Conv2D case
            self._X_padded = np.concatenate(
                    (np.zeros((self._window_size_samples-1, self._X.shape[1])), self._X),
                    axis=0)     # Padding with zeros
            self._X_buf = np.zeros(
                    [self.batch_size, 1, self._window_size_samples, self._X.shape[1]],
                    np.float32)   # Pre-allocating buffers
        elif self._num_conv_dims==1:   # Conv1D case
            self._X_padded = np.concatenate((np.zeros((self._window_size_samples-1, self._X.shape[1])),
                    data), axis=0)     # Padding with zeros
            self._X_buf = np.zeros(
                    [self.batch_size, self._X.shape[1], self._window_size_samples],
                    np.float32)   # Pre-allocating buffers
        else:
            print TAG, 'Wrong num_conv_dims:', num_conv_dims
            
        print TAG, 'self._X_padded size:', self._X_padded.shape
        print TAG, 'self._X_buf size:', self._X_buf.shape

        self._y_buf = np.zeros([self.batch_size, num_outputs], np.float32)

        print TAG, 'self._X size:', self._X.shape
        print TAG, 'self._X_padded size:', self._X_padded.shape

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
                    i_time = np.random.randint((self._window_size_samples-1), self._X.shape[1])

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
        elif self._num_conv_dims == 1:   # Conv1D case
            for i_index, i_time in enumerate(X_indices):
                if i_time < 0:
                    i_time = np.random.randint((self._window_size_samples-1), self._X.shape[0])
                X_window = self._X_padded[i_time:i_time+self._window_size_samples]
                #print(TAG, "X_window part:\n", X_window[0, 2:6, 20:25])
                #print("X_window size:", X_window.shape)
                #print 'self._window_size_samples:', self._window_size_samples
                X_batch[i_index] = TimeSeriesBatchIterator.create_X_instance(
                        X_window, num_conv_dims=1)
                #print 'X_batch.shape:', X_batch.shape
                #X_batch[i_index] = X_window[::-1][::params.DEC_FACTOR].transpose()      # Reversing to get the latest point after the downsampling
                #print TAG, "X_batch[i_index]:\n", X_batch[i_index]
                if y_indices is not None:
                    #print 'i_index, i_time:', i_index, i_time
                    #print 'y_batch.shape:', y_batch.shape
                    #print 'self._y.shape:', self._y.shape
                    y_batch[i_index] = self._y[i_time]
                    #print 'y_batch.shape:', y_batch.shape
        else:
            print TAG, 'Wrong num_conv_dims:', num_conv_dims

        if y_indices is None:
            y_batch = None

        return X_batch, y_batch
