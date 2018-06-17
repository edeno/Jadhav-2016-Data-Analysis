from collections import namedtuple
from os.path import abspath, dirname, join, pardir

SAMPLING_FREQUENCY = 1500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')

Animal = namedtuple('Animal', {'directory', 'short_name'})
ANIMALS = {
    'HPa': Animal(directory=join(RAW_DATA_DIR, 'HPa_direct'),
                  short_name='HPa'),
    'HPb': Animal(directory=join(RAW_DATA_DIR, 'HPb_direct'),
                  short_name='HPb'),
    'HPc': Animal(directory=join(RAW_DATA_DIR, 'HPc_direct'),
                  short_name='HPc'),
}


# Multitaper Parameters
_50Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.020,
    time_window_step=0.020,
    time_halfbandwidth_product=1,
)
_20Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.050,
    time_window_step=0.050,
    time_halfbandwidth_product=1,
)
_10Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.100,
    time_window_step=0.100,
    time_halfbandwidth_product=1,
)
_4Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.250,
    time_window_step=0.250,
    time_halfbandwidth_product=1,
)
_2Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.500,
    time_window_step=0.500,
    time_halfbandwidth_product=1,
)

_12Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.250,
    time_window_step=0.250,
    time_halfbandwidth_product=3,
)

MULTITAPER_PARAMETERS = {
    '12Hz_Resolution': _12Hz_Res,
}


REPLAY_COVARIATES = ['session_time', 'replay_task',
                     'replay_order', 'replay_motion']

FREQUENCY_BANDS = {
    'theta': (4, 12),
    'beta': (12, 30),
    'slow_gamma': (30, 60),
    'mid_gamma': (60, 100),
    'fast_gamma': (100, 125),
    'ripple': (150, 250)
}

ALPHA = 1E-2
