import utils
import params
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import datetime


TAG = '[decimate]'


if __name__ == '__main__':

    # Init logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Select the file to decimate
    data_path = '/home/user/Downloads/storage-double/OITI_2016/wlgy/'
    #data_filename_base = 'wlgy_rt11'
    data_filename_base = 'wlgy_rt12'
    data_filename = data_path + '{}.bdf'.format(data_filename_base)

    # Init values
    freq_decimated = 128.0
    decimation_factor = 2048.0 / freq_decimated
    logging.debug('%s decimation_factor: %f', TAG, decimation_factor)

    logging.debug('%s Timestamp: %s. Loading the file...', TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

    # Load and decimate the file
    X_decimated, labels = utils.load_data_bdf(data_filename, decimation_factor, num_cores=4)

    logging.debug('%s Timestamp: %s. Concatenating the arrays...', TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

    # Create the data table to be written out
    time_axis = np.arange(X_decimated.shape[0]).reshape(X_decimated.shape[0], 1)
    data = np.concatenate((time_axis, X_decimated, labels), axis=1)

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

    logging.debug('%s Timestamp: %s. Saving the data to file...', TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

    # Save the decimated data to file
    header_channels = [
            u'A1', u'A2', u'A3', u'A4', u'A5', u'A6', u'A7', u'A8', u'A9', u'A10', u'A11', u'A12', u'A13', u'A14', u'A15', u'A16',
            u'A17', u'A18', u'A19', u'A20', u'A21', u'A22', u'A23', u'A24', u'A25', u'A26', u'A27', u'A28', u'A29', u'A30', u'A31', u'A32',
            u'B1', u'B2', u'B3', u'B4', u'B5', u'B6', u'B7', u'B8', u'B9', u'B10', u'B11', u'B12', u'B13', u'B14', u'B15', u'B16',
            u'B17', u'B18', u'B19', u'B20', u'B21', u'B22', u'B23', u'B24', u'B25', u'B26', u'B27', u'B28', u'B29', u'B30', u'B31', u'B32',
            u'C1', u'C2', u'C3', u'C4', u'C5', u'C6', u'C7', u'C8', u'C9', u'C10', u'C11', u'C12', u'C13', u'C14', u'C15', u'C16',
            u'C17', u'C18', u'C19', u'C20', u'C21', u'C22', u'C23', u'C24', u'C25', u'C26', u'C27', u'C28', u'C29', u'C30', u'C31', u'C32',
            u'D1', u'D2', u'D3', u'D4', u'D5', u'D6', u'D7', u'D8', u'D9', u'D10', u'D11', u'D12', u'D13', u'D14', u'D15', u'D16',
            u'D17', u'D18', u'D19', u'D20', u'D21', u'D22', u'D23', u'D24', u'D25', u'D26', u'D27', u'D28', u'D29', u'D30', u'D31', u'D32',
            u'EXG1', u'EXG2', u'EXG3', u'EXG4', u'EXG5', u'EXG6', u'EXG7', u'EXG8', u'Status']
    header_list = []
    header_list.append('time')
    header_list.extend(header_channels)
    header = ','.join(header_list)
    #logging.debug('Header list: %s', header_list)
    logging.debug('header: %s', header)
    np.savetxt(data_path + '{0}_{1}Hz.csv'.format(data_filename_base, int(freq_decimated)),
            #'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\{0}\\{1}_decimated_{2}Hz.csv'
            #        .format(data_folder_name, data_filename_base_list[i_filename], int(freq_decimated)),
            X=data, fmt='%.9f', delimiter=",",
            header=header, comments='')

    logging.debug('%s Timestamp: %s', TAG, datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
