from collections import namedtuple

SAMPLING_FREQUENCY = 1500

Animal = namedtuple('Animal', {'directory', 'short_name'})
ANIMALS = {
    'HPa': Animal(directory='HPa_direct', short_name='HPa'),
    'HPb': Animal(directory='HPb_direct', short_name='HPb'),
    'HPc': Animal(directory='HPc_direct', short_name='HPc')
}


# Multitaper Parameters

# Ripple Frequencies
# Frequency Resolution: 50 Hz
ripple_frequency_highTimeRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.020,
    time_window_step=0.020,
    desired_frequencies=(100, 300),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.420, 0.400)
)
# Frequency Resolution: 20 Hz
ripple_frequency_highFreqRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.050,
    time_window_step=0.050,
    desired_frequencies=(30, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.450, 0.500)
)
# Gamma Frequencies
# Frequency Resolution: 20 Hz
gamma_frequency_highTimeRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.050,
    time_window_step=0.050,
    desired_frequencies=(30, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.450, 0.500)
)
# Frequency Resolution: 10 Hz
gamma_frequency_medFreqRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.100,
    time_window_step=0.100,
    desired_frequencies=(30, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.500, 0.500)
)
# Frequency Resolution: 5 Hz
gamma_frequency_highFreqRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.200,
    time_window_step=0.200,
    desired_frequencies=(30, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.600, 0.400)
)

# Low Frequencies
# Frequency Resolution: 10 Hz
low_frequency_highTimeRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.100,
    time_window_step=0.100,
    desired_frequencies=(4, 30),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.500, 0.500)
)
# Frequency Resolution: 4 Hz
low_frequency_medFreqRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.250,
    time_window_step=0.250,
    desired_frequencies=(4, 30),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.750, 0.500)
)
# Frequency Resolution: 2 Hz
low_frequency_highFreqRes = dict(
    sampling_frequency=SAMPLING_FREQUENCY,
    time_window_duration=0.500,
    time_window_step=0.500,
    desired_frequencies=(4, 30),
    time_halfbandwidth_product=1,
    window_of_interest=(-1.00, 0.500)
)

MULTITAPER_PARAMETERS = {
    'ripple_frequencies_50Hz_Res': ripple_frequency_highTimeRes,
    'ripple_frequencies_20Hz_Res': ripple_frequency_highFreqRes,
    'gamma_frequencies_20Hz_Res': gamma_frequency_highTimeRes,
    'gamma_frequencies_10Hz_Res': gamma_frequency_medFreqRes,
    'gamma_frequencies_5Hz_Res': gamma_frequency_highFreqRes,
    'low_frequencies_10Hz_Res': low_frequency_highTimeRes,
    'low_frequencies_4Hz_Res': low_frequency_medFreqRes,
    'low_frequencies_2Hz_Res': low_frequency_highFreqRes
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
