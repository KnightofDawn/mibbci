

import nnfactory
import nnutils
from timeseriesbatchiterator import TimeSeriesBatchIterator
import utils
import params
import numpy as np
import matplotlib.pyplot as plt
import logging


TAG = '[timeseriesprocessor]'


class TimeSeriesProcessor:

    def __init__(self, filename_train, filename_test,
            signal_col_ids, label_col_ids,
            event_name_list,
            #num_channels, num_event_types,
            freq_sampling, decimation_factor,
            freq_cut_lo, freq_cut_hi, M_fir,
            artifact_threshold,
            window_size_decimated_in_samples,
            filename_pipeline_base, nn_type,
            num_max_training_epochs, num_train_data_instances,
            plot=False, runtest=False):

        self._filename_train = filename_train
        self._filename_test = filename_test
        self._event_name_list = event_name_list
        self._signal_col_ids = signal_col_ids
        self._label_col_ids = label_col_ids
        self._num_channels = len(self._signal_col_ids)
        self._num_event_types = len(self._label_col_ids)
        #self._num_channels = num_channels
        #self._num_event_types = num_event_types
        self._freq_sampling = freq_sampling
        self._decimation_factor = decimation_factor
        self._freq_cut_lo = freq_cut_lo
        self._freq_cut_hi = freq_cut_hi
        self._M_fir = M_fir
        self._artifact_threshold = artifact_threshold
        self._window_size_decimated_in_samples = window_size_decimated_in_samples
        #
        self._filename_pipeline_base = filename_pipeline_base
        self._nn_type = nn_type
        self._num_max_training_epochs = num_max_training_epochs
        self._num_train_data_instances = num_train_data_instances

        self._is_plot_mode_on = plot
        self._is_runtest_mode_on = runtest

        logging.debug('%s self._signal_col_ids: %s', TAG, str(self._signal_col_ids))
        logging.debug('%s self._label_col_ids: %s', TAG, str(self._label_col_ids))
        logging.debug('%s self._num_channels: %d', TAG, self._num_channels)
        logging.debug('%s self._num_event_types: %d', TAG, self._num_event_types)


    # End of __init__

    def run(self, load_pipeline=False, train_more=False):

        # Load/pretrain the net
        if load_pipeline:

            # Load the processing pipeline
            nn_input_shape = self.get_nn_input_shape()
            nnet, numer, denom, scaler = utils.load_processing_pipeline(
                    self._filename_pipeline_base,
                    nn_type=self._nn_type,
                    nn_input_shape=nn_input_shape,
                    nn_output_shape=self._num_event_types,
                    num_max_training_epochs=self._num_max_training_epochs)

            if train_more:

                # Load the training data
                X_train_raw, labels_train = utils.load_data(
                        data_filename=self._filename_train,
                        signal_col_ids=self._signal_col_ids,
                        label_col_ids=self._label_col_ids,
                        decimation_factor=self._decimation_factor)

                # Preprocess the data
                numer, denom, scaler = utils.init_preprocessors(
                        X_raw=X_train_raw,
                        freq_sampling=self._freq_sampling,
                        freq_cut_lo=self._freq_cut_lo,
                        freq_cut_hi=self._freq_cut_hi,
                        M_fir=self._M_fir,
                        artifact_threshold=self._artifact_threshold)
                X_train_preproc, labels_train = utils.preprocess(
                        X_train_raw, labels_train,
                        tdfilt_numer=numer, tdfilt_denom=denom,
                        # reref_channel_id=params.REREF_CHANNEL_ID,
                        artifact_threshold=self._artifact_threshold,
                        # power=True,
                        # mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                        scaler=scaler,
                        window_size=self._window_size_decimated_in_samples,
                        nn_type=self._nn_type)

                # Train the NN
                nnet = nnutils.train_nn_from_timeseries(
                        nnet, self._nn_type,
                        X_train_preproc, labels_train,
                        self._window_size_decimated_in_samples,
                        self._num_event_types,
                        self._num_train_data_instances,
                        plot_history=False)

                # Save the pipeline
                utils.save_processing_pipeline(
                        nnet, self._nn_type, numer, denom, scaler)

        else:   # If a new net is to be created

            # Load the training data
            logging.debug('%s Loading the training data...', TAG)
            X_train_raw, labels_train = utils.load_data(
                    data_filename=self._filename_train,
                    signal_col_ids=self._signal_col_ids,
                    label_col_ids=self._label_col_ids,
                    decimation_factor=self._decimation_factor)
            logging.debug('%s X_train_raw.shape: %s', TAG, str(X_train_raw.shape))
            logging.debug('%s labels_train.shape: %s', TAG, str(labels_train.shape))
            logging.debug('%s np.sum(labels_train, axis=0): %s', TAG, str(np.sum(labels_train, axis=0).tolist()))
            logging.debug('%s Training data loaded.', TAG)

            # Preprocess the data
            numer, denom, scaler = utils.init_preprocessors(
                    X_train_raw,
                    self._freq_sampling,
                    self._freq_cut_lo,
                    self._freq_cut_hi,
                    self._M_fir,
                    artifact_threshold=self._artifact_threshold,
                    plot=False)
            X_train_preproc, labels_train = utils.preprocess(
                    X_train_raw, labels_train,
                    tdfilt_numer=numer, tdfilt_denom=denom,
                    # reref_channel_id=params.REREF_CHANNEL_ID,
                    artifact_threshold=self._artifact_threshold,
                    # power=True,
                    # mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                    scaler=scaler,
                    window_size=self._window_size_decimated_in_samples,
                    nn_type=self._nn_type)
            # labels_train = labels_train

            # Plot the training data
            if self._is_plot_mode_on:
                logging.debug('%s Plotting the preprocessed training data... %s', TAG, self._nn_type)
                time_axis = np.arange(X_train_preproc.shape[0])
                if 'gtec' in self._nn_type:
                    t_from = 0
                    t_to = t_from + 120 * self._freq_sampling
                    plot_cols = range(16)
                    #plot_cols = (1, 3, 9)
                    #plt.plot(time_axis[t_from:t_to], X_train_preproc[t_from:t_to, plot_rows, plot_cols], label='tdfilt')
                    #plt.plot(time_axis[t_from:t_to], X_train_raw[t_from:t_to, plot_cols], label='raw')
                    plt.plot(time_axis[t_from:t_to], X_train_preproc[t_from:t_to, plot_cols], label='tdfilt')
                    plt.plot(time_axis[t_from:t_to], 10.0*labels_train[t_from:t_to], linewidth=3, label='event')
                elif 'biosemi' in self._nn_type:
                    t_from = 20000
                    t_to = t_from + 1000 * self._freq_sampling
                    #plot_rows = (6)
                    #plot_cols = (0, 1, 2, 3, 4, 5, 6, 7)
                    #plt.plot(time_axis[t_from:t_to], X_train_preproc[t_from:t_to, plot_rows, plot_cols], label='tdfilt')
                    #plt.plot(time_axis[t_from:t_to], X_train_raw[t_from:t_to, plot_cols], label='raw')
                    plt.plot(time_axis[t_from:t_to], X_train_preproc[t_from:t_to, plot_cols], label='tdfilt')
                    plt.plot(time_axis[t_from:t_to], -300000.0*labels_train[t_from:t_to], linewidth=3, label='event')
                elif 'gal' in self._nn_type:
                    t_from = 0
                    t_to = t_from + 1000 * self._freq_sampling
                    #plot_rows = (6)
                    plot_cols = (0, 1, 2, 3, 4, 5, 6, 7)
                    #plt.plot(time_axis[t_from:t_to], X_train_preproc[t_from:t_to, plot_rows, plot_cols], label='tdfilt')
                    #plt.plot(time_axis[t_from:t_to], X_train_raw[t_from:t_to, plot_cols], label='raw')
                    plt.plot(time_axis[t_from:t_to], X_train_preproc[t_from:t_to, plot_cols], label='tdfilt')
                    plt.plot(time_axis[t_from:t_to], 10.0*labels_train[t_from:t_to], linewidth=3, label='event')
                else:
                    logging.critical('%s Unknown source make.', TAG)

                plt.legend(loc='lower right')
                plt.show()

            # Init the NN
            nn_input_shape = self.get_nn_input_shape()
            nnet, _ = nnfactory.create_nn(
                    nn_type=self._nn_type,
                    nn_input_shape=nn_input_shape,
                    nn_output_shape=self._num_event_types,
                    num_max_training_epochs=self._num_max_training_epochs)

            # Train the NN
            logging.debug('%s Training the NN...', TAG)
            nnet = nnutils.train_nn_from_timeseries(
                    nnet, self._nn_type,
                    X_train_preproc, labels_train,
                    self._window_size_decimated_in_samples,
                    self._num_event_types,
                    self._num_train_data_instances,
                    plot_history=False)
            logging.debug('%s Training the NN finished.', TAG)

            # Save the pipeline
            utils.save_processing_pipeline(
                    nnet, self._nn_type,
                    numer, denom, scaler)


        # Load the test data
        logging.debug('%s Loading the test data...', TAG)
        X_test_raw, labels_test = utils.load_data(
                data_filename=self._filename_test,
                signal_col_ids=self._signal_col_ids,
                label_col_ids=self._label_col_ids,
                decimation_factor=self._decimation_factor)
        if self._is_runtest_mode_on:
            X_test_raw = X_test_raw[0:X_test_raw.shape[0]/4]
            labels_test = labels_test[0:labels_test.shape[0]/4]

        # Pre-process the test data
        X_test_preproc, labels_test = utils.preprocess(
                X_test_raw, labels_test,
                tdfilt_numer=numer, tdfilt_denom=denom,
                #reref_channel_id=params.REREF_CHANNEL_ID,
                artifact_threshold=self._artifact_threshold,
                #power=True,
                #mov_avg_window_size=params.MOVING_AVG_WINDOW_SIZE_SECS,
                scaler=scaler,
                window_size=self._window_size_decimated_in_samples,
                nn_type=self._nn_type)
        X_test_preproc = X_test_preproc[0:40000, :]
        labels_test = labels_test[0:40000, :]

        # Dummy set for testing
        #X_test_preproc = X_train = np.tile(np.reshape(labels_test[:, 0], (labels_test.shape[0], 1)), [1, params.NUM_CHANNELS])

        # Test the net
        batch_iter_test_valid = TimeSeriesBatchIterator(
                data=X_test_preproc, labels=None,
                nn_type=self._nn_type,
                window_size_samples=self._window_size_decimated_in_samples,
                nn_output_shape=self._num_event_types,
                batch_size=params.BATCH_SIZE)
        nnet.batch_iterator_train = None
        nnet.batch_iterator_test = batch_iter_test_valid
        indices_test = np.arange(X_test_preproc.shape[0])
        logging.debug('%s Testing the net...', TAG)
        utils.log_timestamp()
        predictions = nnet.predict_proba(indices_test)
        utils.log_timestamp()
        logging.debug('%s Predictions size: %d, %d', TAG, predictions.shape[0], predictions.shape[1])
        logging.debug('%s np.sum(predictions): %f', TAG, np.sum(predictions))

        # Find the thresholds
        tpr_targets = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        utils.calculate_auroc(
                labels_test, predictions,
                self._event_name_list,
                tpr_targets,
                self._nn_type,
                plot=self._is_plot_mode_on)
        p_thresholds = (0.7, 0.7, 0.7)
        logging.debug('%s p_thresholds: %f, %f, %f', TAG, p_thresholds[0], p_thresholds[1], p_thresholds[2])

        #
        if False:
            create_control_signal(labels_test, predictions, p_thresholds);

    # End of run

    def get_nn_input_shape(self):
        if 'RC' in self._nn_type:
            nn_input_shape = (None, 1, self._window_size_decimated_in_samples, self._num_channels)
        elif 'LC' in self._nn_type:
            nn_input_shape = (None, 1, self._window_size_decimated_in_samples, self._num_channels)
        elif 'Seq' in self._nn_type:
            nn_input_shape = (None, self._window_size_decimated_in_samples, self._num_channels)
        elif 'CovMat' in self._nn_type:
            nn_input_shape = (None, 1, self._num_channels, self._num_channels)
        elif 'TxC' in self._nn_type:
            nn_input_shape = (None, 1, self._window_size_decimated_in_samples, self._num_channels)
        else:
            logging.critical('%s Unknown NN type: %s', TAG, self._nn_type)

        return nn_input_shape

    @staticmethod
    def create_control_signal(labels_test, predictions, p_thresholds):
        # Create the control signal time series
        # p_threshold_rh = 0.754
        # p_threshold_lh = 0.379
        predictions_rh = predictions[:, 0]
        predictions_lh = predictions[:, 1]
        predictions_rh_bin = predictions_rh
        predictions_lh_bin = predictions_lh
        predictions_rh_bin[predictions_rh_bin < p_thresholds[0]] = 0.0
        predictions_rh_bin[predictions_rh_bin >= p_thresholds[0]] = 1.0
        predictions_lh_bin[predictions_lh_bin < p_thresholds[1]] = 0.0
        predictions_lh_bin[predictions_lh_bin >= p_thresholds[1]] = 1.0
        predictions_res_bin = predictions_rh_bin - predictions_lh_bin
        labels_test_res = labels_test[:, 0] - labels_test[:, 1]

        # Create the cursor position time series
        cursor_pos_series = np.zeros(predictions_res_bin.shape)
        for i_time in range(1, predictions_res_bin.shape[0]):
            cursor_pos_series[i_time] = cursor_pos_series[i_time-1] + predictions_res_bin[i_time]

        # Plot prediction confidence histograms
        print 'predictions_rh:\n', predictions_rh
        plt.hist(x=predictions_rh, bins=100)
        plt.title("Right hand")
        plt.show()

        # Plot the timeline
        time_axis = np.arange(labels_test_res.shape[0])
        plt.plot(time_axis, labels_test_res, 'b-', label='event labels')
        plt.plot(time_axis, predictions_res_bin, 'rx', label='predictions')
        plt.plot(time_axis, 0.001*cursor_pos_series, 'r-', label='cursor x pos')
        #plt.plot(time_axis, labels_test[:, 0])
        #plt.plot(time_axis, predictions[:, 0])
        plt.legend(loc='lower right')
        plt.show()
