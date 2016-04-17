

import params
import numpy as np
from nolearn.lasagne import BatchIterator

TAG = '[CustomBatchIterator] '



class EpochBatchIterator(BatchIterator):

    def __init__(self, X_epoch_list, labels=None, *args, **kwargs):
        super(EpochBatchIterator, self).__init__(*args, **kwargs)

        self._X_epoch_list = X_epoch_list
        print 'X_epoch_list[0].shape:', X_epoch_list[0].shape

        if labels is not None:
            self._y = labels
            print TAG, 'len(self._y):', len(self._y)

    # End of init

    @staticmethod
    def create_X_instance(X_window, conv_dim):
        if conv_dim == 1:
            #X_instance = X_window[::-1][::params.DECIMATION_FACTOR_NN].transpose()  # Reversing to get the latest point after the downsampling
            X_instance = X_window.transpose().astype(np.float32)
        elif conv_dim == 2:
            X_instance = X_window.transpose((0, 2, 1)).astype(np.float32)

        return X_instance

    def transform(self, X_epoch_indices, y_indices):
        X_epoch_indices, y_indices = super(EpochBatchIterator, self).transform(X_epoch_indices, y_indices)
        num_indices = len(X_epoch_indices)

        X_batch = np.zeros((num_indices, params.NUM_CHANNELS, int(params.LEN_EPOCH_DECIMATED_SAMPLES)), np.float32)
        y_batch = np.zeros((num_indices, 2), np.float32)

        if False:
            pass
        else:   # Conv1D case
            for i_index, i_epoch in enumerate(X_epoch_indices):
                if i_epoch < 0:
                    i_epoch = np.random.randint(0, len(self._X_epoch_list))

                X_batch[i_index] = EpochBatchIterator.create_X_instance(self._X_epoch_list[i_epoch], conv_dim=1)
                #print 'X_batch.shape:', X_batch.shape
                #print 'type(X_batch):', type(X_batch)
                #print 'X_batch.dtype:', X_batch.dtype
                if y_indices is not None:
                    y_batch[i_index] = self._y[i_epoch]
                    #print 'y_batch.shape:', y_batch.shape
                    #print 'type(y_batch):', type(y_batch)
                    #print 'y_batch.dtype:', y_batch.dtype


        if y_indices is None:
            y_batch = None

        return X_batch, y_batch
