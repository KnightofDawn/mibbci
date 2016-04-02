

import params
import numpy as np
from nolearn.lasagne import BatchIterator

TAG = '[CustomBatchIterator] '



class CustomBatchIterator(BatchIterator):

    def __init__(self, data, labels=None, *args, **kwargs):
        super(CustomBatchIterator, self).__init__(*args, **kwargs)

        self._X = data
        print TAG, 'self._X size:', self._X.shape

        if labels is not None:
            self._y = labels
            print TAG, 'self._y size:', self._y.shape

        if len(self._X.shape) == 3:    # Conv2D case
            self._X_padded = np.concatenate((np.zeros((self._X.shape[0], params.WINDOW_SIZE_RAW_SAMPLES-1, self._X.shape[2])),
                    data), axis=1)     # Padding with zeros
            self._X_buf = np.zeros([self.batch_size, self._X.shape[0], params.NUM_CHANNELS, params.WINDOW_SIZE_DEC_SAMPLES], np.float32)   # Pre-allocating buffers
        else:   # Conv1D case
            self._X_padded = np.concatenate((np.zeros((params.WINDOW_SIZE_RAW_SAMPLES-1, self._X.shape[1])),
                    data), axis=0)     # Padding with zeros
            self._X_buf = np.zeros([self.batch_size, params.NUM_CHANNELS, params.WINDOW_SIZE_DEC_SAMPLES], np.float32)   # Pre-allocating buffers
        print TAG, 'self._X_padded size:', self._X_padded.shape
        print TAG, 'self._X_buf size:', self._X_buf.shape

        self._y_buf = np.zeros([self.batch_size, params.NUM_EVENT_TYPES], np.float32)

        print TAG, 'self._X size:', self._X.shape
        print TAG, 'self._X_padded size:', self._X_padded.shape

    # End of init

    @staticmethod
    def create_X_instance(X_window, conv_dim):
        if conv_dim == 1:
            X_instance = X_window[::-1][::params.DEC_FACTOR].transpose()  # Reversing to get the latest point after the downsampling
        elif conv_dim == 2:
            X_instance = X_window[::-1][:, ::params.DEC_FACTOR].transpose((0, 2, 1))  # Reversing to get the latest point after the downsampling

        return X_instance

    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(CustomBatchIterator, self).transform(X_indices, y_indices)
        #print(TAG, "X_indices part:\n", X_indices[20:30])
        #[count] = X_indices.shape
        #X_batch = self._X_buf[:count]
        #y_batch = self._Y_buf[:count]
        num_indices = X_indices.shape[0]

        X_batch = self._X_buf[:num_indices]
        y_batch = self._y_buf[:num_indices]

        if len(self._X.shape) == 3:     # Conv2D case
            for i_index, i_time in enumerate(X_indices):
                if i_time < 0:
                    i_time = np.random.randint((params.WINDOW_SIZE_RAW_SAMPLES-1), self._X.shape[1])
                X_window = self._X_padded[:, i_time:i_time+params.WINDOW_SIZE_RAW_SAMPLES]
                #print(TAG, "X_window part:\n", X_window[0, 2:6, 20:25])
                #print("X_window size:", X_window.shape)
                X_batch[i_index] = CustomBatchIterator.create_X_instance(X_window, conv_dim=2)
                #X_batch[i_index] = X_window[::-1][:, ::params.DEC_FACTOR].transpose((0, 2, 1))      # Reversing to get the latest point after the downsampling
                #print(TAG, "X_batch[i_index] part:\n", X_batch[i_index, 0, 2:6, 20:25])
                if y_indices is not None:
                    y_batch[i_index] = self._y[i_time]
        else:   # Conv1D case
            for i_index, i_time in enumerate(X_indices):
                if i_time < 0:
                    i_time = np.random.randint((params.WINDOW_SIZE_RAW_SAMPLES-1), self._X.shape[0])
                X_window = self._X_padded[i_time:i_time+params.WINDOW_SIZE_RAW_SAMPLES]
                #print(TAG, "X_window part:\n", X_window[0, 2:6, 20:25])
                #print("X_window size:", X_window.shape)
                X_batch[i_index] = CustomBatchIterator.create_X_instance(X_window, conv_dim=1)
                #X_batch[i_index] = X_window[::-1][::params.DEC_FACTOR].transpose()      # Reversing to get the latest point after the downsampling
                #print(TAG, "X_batch[i_index] part:\n", X_batch[i_index, 0, 2:6, 20:25])
                if y_indices is not None:
                    #print 'i_index, i_time:', i_index, i_time
                    #print 'y_batch.shape:', y_batch.shape
                    #print 'self._y.shape:', self._y.shape
                    y_batch[i_index] = self._y[i_time]


        if y_indices is None:
            y_batch = None

        return X_batch, y_batch
