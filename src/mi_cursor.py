'''

TODO:
- do softmax with so the preds are exclusive
- apply some post-proc i.e. pred rh only after N good preds in a row
- more intensive movements
- try different limbs, try a sportsman who is used to limb control, sportlovo


Older ideas:
- ratio, not difference
- deadband instead of rest state calib



'''
from recorder import Recorder
from timeseriesbatchiterator import TimeSeriesBatchIterator
import params
import nnutils
import graphics
import math
import threading
import numpy as np
import time
from datetime import datetime
from scipy import signal
import matplotlib.pyplot as plt
import cPickle




########################################################################################################################

# 1-16 INDEXING, NOT 0-15: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP5, CPz, CP6, P3, Pz, P4

#CONFIG_FILE_NAME = 'config_test.txt'
is_simulation_mode = False
#is_simulation_mode = True


########################################################################################################################

def get_time_domain_filters(freq_cut_lo, freq_cut_hi, freq_trans):
    freq_Nyq = params.FREQ_S / 2.
    freqs_FIR_Hz = np.array([freq_cut_lo - freq_trans, freq_cut_hi + freq_trans])
    # numer = signal.firwin(M_FIR, freqs_FIR, nyq=FREQ_S/2., pass_zero=False, window="hamming", scale=False)
    numer = signal.firwin(params.M_FIR, freqs_FIR_Hz, nyq=freq_Nyq, pass_zero=False, window="hamming", scale=False)
    denom = 1.
    '''w, h = signal.freqz(numer)
    plt.plot(freq_Nyq*w/math.pi, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()'''


########################################################################################################################

def calc_cursor_step(nnet, X_in):
    print 'X_in.shape:', X_in.shape
    y_out = nnet.get_output(layer='Output', X=X_in)
    print 'y_out.shape:', y_out.shape
    print 'y_out:', y_out
    #lasagne.layers.get_output(nnet_last_layer, inputs=x_in)
    cursor_step = params.CURSOR_STEP_MULT * (y_out[0, 0] - y_out[0, 1])

    return cursor_step


########################################################################################################################

def cursor_func():

    print 'cursor_func(.) entered.'
    cursor_radius = 26
    w = 2 * math.pi / 10

    # Initialize the time-domain filter
    #numer, denom = get_time_domain_filters(8.0, 12.0, 0.5)

    # Init the NN
    filename_base = '../models/MIBBCI_NN_medium_bestsofar'
    filename_nn = filename_base + '.npz'
    nnet = nnutils.load_nn(nnutils.create_nn_medium, filename_nn)

    # Init the preproc stuff
    filename_p = filename_base + '.p'
    scaler = cPickle.load(open(filename_p, 'rb'))
    print 'Loaded scaler.mean_, scaler.var_:', scaler.mean_, scaler.var_

    # Init graphics
    win = graphics.GraphWin('Cursor', params.IMAGE_W, params.IMAGE_H)
    cursor = graphics.Circle(graphics.Point(params.IMAGE_W/2, params.IMAGE_H/2), cursor_radius)
    cursor.setFill(graphics.color_rgb(params.CURSOR_COLOR_REST[0], params.CURSOR_COLOR_REST[1], params.CURSOR_COLOR_REST[2]))
    cursor.setOutline(graphics.color_rgb(params.CURSOR_COLOR_REST[0], params.CURSOR_COLOR_REST[1], params.CURSOR_COLOR_REST[2]))
    cursor.draw(win)
    cursor_pos_prev = np.array([params.IMAGE_W/2, params.IMAGE_H/2])
    cursor_pos = cursor_pos_prev

    # Init event labels
    event_arr_right = np.zeros((params.LEN_DATA_CHUNK_READ, params.NUM_EVENT_TYPES))
    event_arr_right[:, params.EVENT_ID_RH] = np.ones(params.LEN_DATA_CHUNK_READ)
    event_arr_left = np.zeros((params.LEN_DATA_CHUNK_READ, params.NUM_EVENT_TYPES))
    event_arr_left[:, params.EVENT_ID_LH] = np.ones(params.LEN_DATA_CHUNK_READ)
    event_arr_idle = np.zeros((params.LEN_DATA_CHUNK_READ, params.NUM_EVENT_TYPES))
    event_arr_idle[:, params.EVENT_ID_IDLE] = np.ones(params.LEN_DATA_CHUNK_READ)
    #event_arr_calib = np.zeros((params.LEN_DATA_CHUNK_READ, params.NUM_EVENT_TYPES))
    #event_arr_calib[:, 3] = np.ones(params.LEN_DATA_CHUNK_READ)
    cursor_event_list = []
    cursor_color_arr_raw = np.zeros((int(params.LEN_PERIOD_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ), 3))
    color_counter = 0
    for i in range(int(params.LEN_IDLE_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ)):
        cursor_color_arr_raw[color_counter, :] = params.CURSOR_COLOR_IDLE
        cursor_event_list.append(event_arr_idle)      # r, l, idle, calib
        color_counter += 1
    for i in range(int(params.LEN_RIGHT_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ)):
        cursor_color_arr_raw[color_counter, :] = params.CURSOR_COLOR_RIGHT
        cursor_event_list.append(event_arr_right)
        color_counter += 1
    for i in range(int(params.LEN_IDLE_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ)):
        cursor_color_arr_raw[color_counter, :] = params.CURSOR_COLOR_IDLE
        cursor_event_list.append(event_arr_idle)
        color_counter += 1
    for i in range(int(params.LEN_LEFT_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ)):
        cursor_color_arr_raw[color_counter, :] = params.CURSOR_COLOR_LEFT
        cursor_event_list.append(event_arr_left)
        color_counter += 1
    conv_window = np.ones((params.LEN_COLOR_CONV_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ, 1))\
                  / (1 * int(params.LEN_COLOR_CONV_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ))
    cursor_color_arr_ud = np.flipud(cursor_color_arr_raw)
    cursor_color_arr_ud_convd = signal.convolve(cursor_color_arr_ud.T, conv_window.T).T
    cursor_color_arr_final = np.flipud(cursor_color_arr_ud_convd[0:cursor_color_arr_raw.shape[0], :])
    if False:
        plt.figure()
        plt.plot(cursor_color_arr_raw)
        #plt.plot(cursor_color_arr_ud[:, 0])
        #plt.plot(cursor_color_arr_ud_convd[:, 0])
        plt.plot(cursor_color_arr_final)
        #plt.legend(['raw', 'ud', 'ud_convd', 'final'])
        plt.show()

    # Initialize the amplifier
    if not is_simulation_mode:
        print 'Initializing the amp...'
        recorder = Recorder('lslamp', params.FREQ_S, params.LEN_REC_BUF_SEC, params.NUM_CHANNELS)
        thread_rec = threading.Thread(target=recorder.record)
        thread_rec.start()

    # Cursor control loop
    X_raw_buf_live = np.zeros((int(params.FREQ_S*params.LEN_REC_BUF_SEC), params.NUM_CHANNELS))
    label_buf_live = np.zeros((int(params.FREQ_S*params.LEN_REC_BUF_SEC), params.NUM_EVENT_TYPES))
    counter = 0
    #while True:
    while counter < (params.LEN_REC_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ):
        print 'counter: ', counter

        # Clear the canvas
        win.delete('all')

        if not is_simulation_mode:
            # Wait for new data and get it
            data_last_chunk = recorder.get_new_data(params.LEN_DATA_CHUNK_READ, params.AMP_WAIT_SEC)
            recorder.acknowledge_new_data()
            print 'recorder.new_data_counter:', recorder.new_data_counter
        else:
            time.sleep(1.0 / (params.FREQ_S/params.LEN_DATA_CHUNK_READ))
            data_last_chunk = 1000.0 * np.random.rand(int(params.LEN_DATA_CHUNK_READ), params.NUM_CHANNELS)
            #print 'Random data_last_chunk size:', data_last_chunk

        # Insert the new sample into our time series
        i_row_lb = int((counter+params.LEN_PADDING)*params.LEN_DATA_CHUNK_READ)
        i_row_ub = int((counter+params.LEN_PADDING+1)*params.LEN_DATA_CHUNK_READ)
        X_raw_buf_live[i_row_lb:i_row_ub, :] = data_last_chunk
        #print 'data_last_chunk:', data_last_chunk
        label_buf_live[i_row_lb:i_row_ub, :]\
                = cursor_event_list[counter % int(params.LEN_PERIOD_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ)]

        # Calculating cursor step
        i_row_ub = int((counter+params.LEN_PADDING+1)*params.LEN_DATA_CHUNK_READ)
        i_row_lb = i_row_ub - int(params.WINDOW_SIZE_RAW_SAMPLES)
        if i_row_lb >= 0:
            #print 'i_row_lb, i_row_ub:', i_row_lb, i_row_ub
            #print 'X_raw_buf_live[i_row_lb:i_row_ub, :].shape:', X_raw_buf_live[i_row_lb:i_row_ub, :].shape
            X_window = nnutils.preprocess(X_raw_buf_live[i_row_lb:i_row_ub, :], scaler)
            X_in = TimeSeriesBatchIterator.create_X_instance(X_window, conv_dim=1)
            X_in = X_in.reshape(1, X_in.shape[0], X_in.shape[1])
            #print 'X_window.shape:', X_window.shape
            #print 'X_in.shape:', X_in.shape
            cursor_step = calc_cursor_step(nnet, X_in.astype(np.float32))
            cursor_pos = cursor_pos_prev + np.array([cursor_step, 0])
            #print 'cursor_pos: ', cursor_pos
        else:
            cursor_pos = cursor_pos_prev

        cursor_pos_point = graphics.Point(cursor_pos[0], cursor_pos[1])
        cursor_pos_prev = cursor_pos
        cursor = graphics.Circle(cursor_pos_point, cursor_radius)
        color_temp = cursor_color_arr_final[counter % int(params.LEN_PERIOD_SEC * params.FREQ_S / params.LEN_DATA_CHUNK_READ)]
        cursor.setFill(graphics.color_rgb(color_temp[0], color_temp[1], color_temp[2]))
        cursor.setOutline(graphics.color_rgb(color_temp[0], color_temp[1], color_temp[2]))
        cursor.draw(win)

        counter += 1

        # End of if
    # End of while

    # Stop recording
    recorder.stop_recording()

    # Close the window
    win.close()

    # Cut the padding from the data
    i_row_lb = int(params.LEN_PADDING * params.LEN_DATA_CHUNK_READ)
    i_row_ub = int((counter+params.LEN_PADDING)*params.LEN_DATA_CHUNK_READ)
    X_raw_buf_cut = X_raw_buf_live[i_row_lb:i_row_ub, :]
    label_buf_cut = label_buf_live[i_row_lb:i_row_ub, :]

    # Save data to file
    time_axis = np.arange(X_raw_buf_cut.shape[0]).reshape((X_raw_buf_cut.shape[0], 1))
    print 'time_axis.shape:', time_axis.shape
    data_merged = np.concatenate((time_axis, X_raw_buf_cut, label_buf_cut), axis=1)
    print 'data_merged.shape: ', data_merged.shape
    time_save = datetime.now()
    np.savetxt('../data/MIBBCI_REC_{0}{1:02}{2:02}_{3:02}h{4:02}m{5:02}s_RAW.csv'.format(time_save.year, time_save.month, time_save.day,
               time_save.hour, time_save.minute, time_save.second),
               X=data_merged, fmt='%.8f', delimiter=",",
               header='time, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, red, blue, idle',
               comments='')


    print 'cursor_func(.) terminates.'




########################################################################################################################
#
#    MAIN
#
########################################################################################################################


if __name__ == '__main__':

    print 'Main started.'


    # Start the threads
    #thread_rec = threading.Thread(target=recorder.record)
    #thread_cursor = threading.Thread(target=cursor_func)
    #thread_rec.start()
    #thread_cursor.start()
    cursor_func()




    print 'Main terminates.'
