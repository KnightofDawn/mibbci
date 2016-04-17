import utils
import params
import numpy as np
import sys
import logging
import datetime




if __name__ == '__main__':

    # Init logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Parse command line
    if len(sys.argv) == 3:
        data_path = sys.argv[1]
        data_filename = sys.argv[2]
    else:
        logging.error('Incorrect command line arguments.')
        sys.exit()

    # Init values
    freq_decimated = 128.0
    decimation_factor = 2048.0 / freq_decimated
    logging.debug('decimation_factor: %f', decimation_factor)

    # Select the files to decimate
    #data_folder_name = 'Emri_Akos_20160330'
    #data_folder_name = 'Pap_Henrik_20160404'
    #data_folder_name = 'Varga_Peter_20160330'
    data_filename_base_list = []
    data_filename_base_list.append(data_filename)
    #data_filename_base_list.append('vape_rc_12')
    data_filename_list = []
    for filename_base in data_filename_base_list:
        data_filename_list.append(data_path + data_filename)
                #'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\{0}\\{1}.bdf'.format(data_folder_name, filename_base))

    # Loop over the files
    for i_filename in range(len(data_filename_list)):

        logging.debug('Timestamp: %s. Loading the next file...', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

        # Load and decimate the file
        data_filename_list_temp = []
        data_filename_list_temp.append(data_filename_list[i_filename])
        X_decimated, labels = utils.load_data_bdf(data_filename_list_temp, decimation_factor, num_cores=4)

        logging.debug('Timestamp: %s.  Concatenating the arrays...', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

        # Create the data table to be written out
        time_axis = np.arange(X_decimated.shape[0]).reshape(X_decimated.shape[0], 1)
        data = np.concatenate((time_axis, X_decimated, labels), axis=1)

        logging.debug('Timestamp: %s. Saving the data to file...', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

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
        data_filename_tokens = data_filename.split('.')
        if len(data_filename_tokens) > 2:
            data_filename_base = '.'.join(data_filename_tokens[0:-2])
        else:
            data_filename_base = data_filename_tokens[0]
        logging.debug('data_filename_base: %s', data_filename_base)
        np.savetxt(data_path + '{0}_decimated_{1}Hz.csv'.format(data_filename_base, int(freq_decimated)),
                #'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\{0}\\{1}_decimated_{2}Hz.csv'
                #        .format(data_folder_name, data_filename_base_list[i_filename], int(freq_decimated)),
                X=data, fmt='%.9f', delimiter=",",
                header=header, comments='')

        logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))

    # End of loop over filenames
