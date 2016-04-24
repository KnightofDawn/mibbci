

import numpy as np
import lasagne



TIMESTAMP_FORMAT_STR = '%Y-%m-%d %H:%M:%S'



# EEG setup-related
CHANNEL_NAMES_GTEC = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CPz', 'CP6', 'P3', 'Pz', 'P4']
EVENT_NAMES_GTEC = ['rh', 'lh', 'idle']
CHANNEL_NAMES_BIOSEMI = [
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16',
        'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32',
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16',
        'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32',
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
        'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32',
        'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16',
        'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32',
        'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
EVENT_NAMES_BIOSEMI = ['dn']
IMAGE_SIZE_BDF = (11, 24)   # rows, cols
FREQ_S_CSV = 128.0
FREQ_S_BDF = 2048.0
#NUM_CHANNELS_CSV = 128 #16
NUM_CHANNELS_BDF = 128
LEN_INITIAL_DATA = 128
AMP_WAIT_SEC = 0.01
LEN_DATA_CHUNK_READ = 16.0
LEN_RIGHT_SEC = 5.0
LEN_LEFT_SEC = 5.0
LEN_IDLE_SEC = 5.0
LEN_PERIOD_SEC = LEN_IDLE_SEC + LEN_RIGHT_SEC + LEN_IDLE_SEC + LEN_LEFT_SEC
LEN_COLOR_CONV_SEC = 1.0
LEN_CALIB_SEC = LEN_PERIOD_SEC
LEN_REC_SEC = 10.0 * LEN_PERIOD_SEC
LEN_REC_BUF_SEC = 1.2 * LEN_REC_SEC


# Paradigm-related
#NUM_EVENT_TYPES_CSV = 3     # r, l, idle
EVENT_ID_RH = 0
EVENT_ID_LH = 1
EVENT_ID_IDLE = 2
LABEL_ID_RED = 17
NUM_EVENT_TYPES_BDF = 1     # button down
EVENT_LENGTH_SAMPLES = 512
FEAT_MULT_1_SIMU = 0.002
FEAT_MULT_2_SIMU = 1.0
FEAT_MULT_1_REAL = 0.001
FEAT_MULT_2_REAL = 1.0
IMAGE_W = 960
IMAGE_H = 640
#CURSOR_COLOR_REST = 'green'
#CURSOR_COLOR_RIGHT = 'red'
#CURSOR_COLOR_LEFT = 'blue'
#CURSOR_COLOR_IDLE = 'black'
CURSOR_COLOR_REST = np.array([0, 127, 0])    # RGB
CURSOR_COLOR_RIGHT = np.array([255, 0, 0])
CURSOR_COLOR_LEFT = np.array([0, 0, 255])
CURSOR_COLOR_IDLE = np.array([0, 0, 0])


# Signal proc
#LEN_EPOCH_DECIMATED_SAMPLES = int(4.0 * FREQ_S / DECIMATION_FACTOR_PREPROC)
#EPOCH_OFFSET_SAMPLES = -int(2.0 * FREQ_S / DECIMATION_FACTOR_PREPROC)
NUM_PARALLEL_JOBS = 4


# NN
BATCH_SIZE = 32


# Cursor
CURSOR_STEP_MULT = 0.0
