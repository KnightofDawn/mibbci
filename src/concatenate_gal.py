import utils
import params
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import datetime


TAG = '[concatenate_gal]'


if __name__ == '__main__':

    # Init logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Init values
    freq_decimated = 500.0
    num_channels = 32
    signal_col_ids = range(1, (1 + num_channels))
    num_event_types = 6
    label_col_ids = range(1, (1 + num_event_types))
    channel_names = params.CHANNEL_NAMES_GAL
    event_names = params.EVENT_NAMES_GAL

    # Select the files to concatenate
    data_path = '/home/user/Downloads/storage-double/Kaggle_GAL_data/train/'
    data_filename_data_list = []
    data_filename_events_list = []
    subj_from = 1
    subj_to = 1
    series_from = 1
    series_to = 2

    # Loop over subjects
    for i_subj in range(subj_from, (subj_to+1)):

        for i_series in range(series_from, (series_to+1)):
            data_filename_data_list.append(data_path + 'subj{0}_series{1}_data.csv'.format(i_subj, i_series))
            data_filename_events_list.append(data_path + 'subj{0}_series{1}_events.csv'.format(i_subj, i_series))

        data_filename_concatenated = 'subj{0}-{1}_series{2}-{3}_RAW.csv'.format(subj_from, subj_to, series_from, series_to)

        logging.debug('%s Timestamp: %s. Loading the next file...', TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

        # Load the files
        signal_list = []
        for data_filename_data in data_filename_data_list:
            signal, _ = utils.load_data_csv(
                    data_csv_filename=data_filename_data,
                    signal_col_ids=signal_col_ids,
                    label_col_ids=[],
                    decimation_factor=1.0)
            logging.debug('%s signal.shape: %s', TAG, str(signal.shape))
            signal_list.append(signal)
        labels_list = []
        for data_filename_events in data_filename_events_list:
            _, labels = utils.load_data_csv(
                    data_csv_filename=data_filename_events,
                    signal_col_ids=[],
                    label_col_ids=label_col_ids,
                    decimation_factor=1.0)
            logging.debug('%s labels.shape: %s', TAG, str(labels.shape))
            labels_list.append(labels)

        logging.debug('%s Timestamp: %s. Concatenating the arrays...',
                TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
        signal = np.concatenate(signal_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Create the data table to be written out
        time_axis = np.arange(signal.shape[0]).reshape(signal.shape[0], 1)
        data = np.concatenate((time_axis, signal, labels), axis=1)

        # Plot the data
        if False:
            time_to = 2048*5
            time_axis = np.arange(time_to)
            channels_to_plot = (12, 54, 76, 111)
            #plt.plot(time_axis, X_raw[0:time_to, channels_to_plot], label='raw')
            #plt.plot(time_axis, X_lpfiltered[0:time_to, channels_to_plot], label='tdfilt')
            plt.plot(time_axis, data[0:time_to, -1], label='event')
            plt.title('Decimated data')
            plt.legend(loc='lower right')
            plt.show()

        logging.debug('%s Timestamp: %s. Saving the data to file...',
                TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

        # Save the decimated data to file
        header_list = []
        header_list.append('time')
        header_list.extend(channel_names)
        header_list.extend(event_names)
        header = ','.join(header_list)
        #logging.debug('Header list: %s', header_list)
        logging.debug('header: %s', header)
        np.savetxt(data_path + data_filename_concatenated,
                X=data, fmt='%.9f', delimiter=",",
                header=header, comments='')

        logging.debug('%s Timestamp: %s',
                TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

    # End of loop over subjects
