import utils
import params
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import datetime


TAG = '[concatenate]'


if __name__ == '__main__':

    # Init logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Init values
    freq_decimated = 128.0
    num_channels = 16
    num_event_types = 3
    channel_names = params.CHANNEL_NAMES_GTEC
    event_names = params.EVENT_NAMES_GTEC

    # Select the files to concatenate
    data_path = '/home/user/Downloads/storage-double/gUSBamp_20160402_VK_2/'
    data_filename_list = []
    #data_filename_list.append(data_path + 'MIBBCI_REC_128Hz_20160402_18h57m57s_RAW.csv')
    data_filename_list.append(data_path + 'MIBBCI_REC_128Hz_20160402_19h17m10s_RAW.csv')
    #data_filename_list.append(data_path + 'MIBBCI_REC_128Hz_20160402_19h21m49s_RAW.csv')
    data_filename_list.append(data_path + 'MIBBCI_REC_128Hz_20160402_19h26m01s_RAW.csv')
    data_filename_concatenated = 'MIBBCI_REC_128Hz_20160402_4-6_RAW.csv'

    logging.debug('%s Timestamp: %s. Loading the next file...', TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

    # Load the files
    signal_list = []
    labels_list = []
    for data_filename in data_filename_list:
        signal, labels = utils.load_data_csv(
                data_filename,
                num_channels, num_event_types, 1.0)
        signal_list.append(signal)
        labels_list.append(labels)

    logging.debug('%s Timestamp: %s.  Concatenating the arrays...',
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
