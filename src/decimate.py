import utils
import params
import numpy as np
import sys
import logging
import datetime




if __name__ == '__main__':

    # Init logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Init values
    freq_decimated = 128.0
    decimation_factor = 2048.0 / freq_decimated
    logging.debug('decimation_factor: %f', decimation_factor)

    # Select the files to decimate
    #data_folder_name = 'Emri_Akos_20160330'
    #data_folder_name = 'Pap_Henrik_20160404'
    data_folder_name = 'Varga_Peter_20160330'
    data_filename_base_list = []
    data_filename_base_list.append('rc_11')
    data_filename_list = []
    for filename_base in data_filename_base_list:
        data_filename_list.append(
                'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\{0}\\{1}.bdf'
                        .format(data_folder_name, filename_base))

    # Loop over the files
    for i_filename in range(len(data_filename_list)):
    
        logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
    
        # Load and decimate the file
        data_filename_list_temp = []
        data_filename_list_temp.append(data_filename_list[i_filename])
        X_decimated, labels = utils.load_data_bdf(data_filename_list_temp, decimation_factor)
        
        logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
        
        # Create the data table to be written out
        time_axis = np.arange(X_decimated.shape[0]).reshape(X_decimated.shape[0], 1)
        data = np.concatenate((time_axis, X_decimated, labels), axis=1)
    
        logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
    
        # Save the decimated data to file
        np.savetxt(
                'C:\\Users\\user\\Downloads\\storage_double\\OITI_2016\\{0}\\{1}_decimated_{2}Hz.csv'
                        .format(data_folder_name, data_filename_base_list[i_filename], int(freq_decimated)),
                X=data, fmt='%.9f', delimiter=",")
                #header='time, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15',
                #comments='')
        
        logging.debug('Timestamp: %s', datetime.datetime.now().strftime(params.TIMESTAMP_FORMAT_STR))
        
    # End of loop over filenames
                
                
                