'''

TODO
- IMG - Fill the image holes
- Stop bursts by muting the output after a rh/lh for 0.5 secs

Current bests:
- gtec_CovMat_small
-


http://rosinality.ncity.net/doku.php?id=python:installing_theano

'''

from timeseriesprocessor import TimeSeriesProcessor
import params
import numpy as np
import sys
import logging


TAG = '[process_nn]'




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    # Init logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.debug('Main started.')

    # Command parameters
    is_runtest_mode_on = False
    is_plot_mode_on = True
    is_net_pretrained = False
    is_net_to_train_more = False

    # Set processing pipeline location and attributes
    #nn_type = 'gtec_CovMat_medium'
    #nn_type = 'gtec_CovMat_small'
    #nn_type = 'gtec_Seq_recurrent'
    nn_type = 'gtec_Seq_lstm'
    #nn_type = 'gtec_RC'
    #nn_type = 'gtec_LC'
    #nn_type = 'biosemi_Seq_recurrent'
    #nn_type = 'biosemi_CovMat_medium'
    filename_pipeline_base = './models/MIBBCI_NN_{0}_20160423_21h07m55s'.format(nn_type)

    # Set preliminary values
    if is_runtest_mode_on:
        num_max_training_epochs = 3
        num_train_data_instances = 3
    else:
        num_max_training_epochs = 100
        num_train_data_instances = 1024
    if 'biosemi' in nn_type:
        #
        #filename_train = '/home/user/Downloads/storage-double/OITI_2016/Pap_Henrik_20160404/pahe_rt11_128Hz.csv'
        #filename_train = '/home/user/Downloads/storage-double/OITI_2016/Emri_Akos_20160330/emak_rt11_128Hz.csv'
        filename_train = '/home/user/Downloads/storage-double/OITI_2016/wlgy/wlgy_rt11_128Hz.csv'
        #
        #filename_test = '/home/user/Downloads/storage-double/OITI_2016/Pap_Henrik_20160404/pahe_rt12_128Hz.csv'
        #filename_test = '/home/user/Downloads/storage-double/OITI_2016/Emri_Akos_20160330/emak_rt12_128Hz.csv'
        filename_test = '/home/user/Downloads/storage-double/OITI_2016/wlgy/wlgy_rt12_128Hz.csv'
        #
        num_channels = 128
        signal_col_ids = range(1, (1 + num_channels))
        #num_channels = 8
        #signal_col_ids = [129, 130, 131, 132, 133, 134, 135, 136]   # In the raw csv with the time axis
        label_col_ids = [137]
        freq_sampling = 128.0
        decimation_factor = 1.0
        freq_cut_lo = 2.0
        freq_cut_hi = 40.0
        window_size_decimated_in_samples = int(1.0 * freq_sampling)
        M_fir = int(1.0 * window_size_decimated_in_samples)
        event_name_list = ['btn dn']
    elif 'gtec' in nn_type:
        #
        filename_train = '/home/user/Downloads/storage-double/gUSBamp_20160402_VK_2/MIBBCI_REC_128Hz_20160402_1-2-3-4-5_RAW.csv'
        #
        filename_test = '/home/user/Downloads/storage-double/gUSBamp_20160402_VK_2/MIBBCI_REC_128Hz_20160402_6_RAW.csv'
        #
        num_channels = 16
        signal_col_ids = range(1, (1 + num_channels))
        label_col_ids = [17, 18]
        freq_sampling = 128.0
        decimation_factor = 1.0
        freq_cut_lo = 4.0
        freq_cut_hi = 40.0
        window_size_decimated_in_samples = int(1.0 * freq_sampling)
        M_fir = int(1.0 * window_size_decimated_in_samples)
        artifact_threshold = 25.0
        event_name_list = ['rh', 'lh', 'idle']

    # Init the data processor
    proc = TimeSeriesProcessor(
            filename_train=filename_train,
            filename_test=filename_test,
            signal_col_ids=signal_col_ids,
            label_col_ids=label_col_ids,
            event_name_list=event_name_list,
            #num_channels, num_event_types,
            freq_sampling=freq_sampling,
            decimation_factor=decimation_factor,
            freq_cut_lo=freq_cut_lo,
            freq_cut_hi=freq_cut_hi,
            M_fir=M_fir,
            artifact_threshold=artifact_threshold,
            window_size_decimated_in_samples=window_size_decimated_in_samples,
            filename_pipeline_base=filename_pipeline_base,
            nn_type=nn_type,
            num_max_training_epochs=num_max_training_epochs,
            num_train_data_instances=num_train_data_instances,
            plot=is_plot_mode_on, runtest=is_runtest_mode_on)

    # Run the data processor
    proc.run(load_pipeline=is_net_pretrained, train_more=is_net_to_train_more)


    logging.debug('Main terminates.')
