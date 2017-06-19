from collections import namedtuple

SAMPLING_FREQUENCY = 1500

Animal = namedtuple('Animal', {'directory', 'short_name'})
ANIMALS = {
    'HPa': Animal(directory='HPa_direct', short_name='HPa'),
    'HPb': Animal(directory='HPb_direct', short_name='HPb'),
    'HPc': Animal(directory='HPc_direct', short_name='HPc')
}


# Multitaper Parameters
_50Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.020,
    time_window_step=0.020,
    time_halfbandwidth_product=1,
    window_of_interest=(-0.420, 0.400)
)
_20Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.050,
    time_window_step=0.050,
    time_halfbandwidth_product=1,
    window_of_interest=(-0.450, 0.500)
)
_10Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.100,
    time_window_step=0.100,
    time_halfbandwidth_product=1,
    window_of_interest=(-0.500, 0.500)
)
_4Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.250,
    time_window_step=0.250,
    time_halfbandwidth_product=1,
    window_of_interest=(-0.750, 0.500)
)
_2Hz_Res = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.500,
    time_window_step=0.500,
    time_halfbandwidth_product=1,
    window_of_interest=(-1.00, 0.500)
)

MULTITAPER_PARAMETERS = {
    '50Hz_Resolution': _50Hz_Res,
    '20Hz_Resolution': _20Hz_Res,
    '10Hz_Resolution': _10Hz_Res,
    '4Hz_Resolution': _4Hz_Res,
    '2Hz_Resolution': _2Hz_Res
}


RIPPLE_COVARIATES = ['session_time', 'ripple_trajectory',
                     'ripple_direction']

FREQUENCY_BANDS = {
    'theta': (4, 12),
    'beta': (12, 30),
    'slow_gamma': (30, 60),
    'mid_gamma': (60, 100),
    'fast_gamma': (100, 125),
    'ripple': (150, 250)
}

ALPHA = 1E-2

N_DAYS = 8
