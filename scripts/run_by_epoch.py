'''Exectue set of functions for each epoch
'''
from collections import namedtuple
from datetime import datetime
from subprocess import PIPE, run
from sys import argv, exit

from src.analysis import (canonical_coherence_by_ripple_type,
                          coherence_by_ripple_type,
                          ripple_triggered_canonical_coherence,
                          ripple_triggered_coherence, save_ripple_info)
from src.ripple_decoding import decode_ripple
from src.ripple_detection import get_epoch_ripples

sampling_frequency = 1500
Animal = namedtuple('Animal', {'directory', 'short_name'})
animals = {
    'HPa': Animal(directory='HPa_direct', short_name='HPa'),
    'HPb': Animal(directory='HPb_direct', short_name='HPb'),
    'HPc': Animal(directory='HPc_direct', short_name='HPc')
}
ripple_frequency = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.020,
    time_window_step=0.020,
    desired_frequencies=(100, 300),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.420, 0.400)
)
gamma_frequency_highTimeRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.050,
    time_window_step=0.050,
    desired_frequencies=(15, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.450, 0.400)
)
gamma_frequency_medFreqRes1 = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.100,
    time_window_step=0.100,
    desired_frequencies=(15, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.500, 0.400)
)
gamma_frequency_medFreqRes2 = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.200,
    time_window_step=0.200,
    desired_frequencies=(15, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.600, 0.400)
)
gamma_frequency_highFreqRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.400,
    time_window_step=0.400,
    desired_frequencies=(15, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.800, 0.400)
)
low_frequency_highTimeRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.100,
    time_window_step=0.100,
    desired_frequencies=(0, 20),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.500, 0.400)
)
low_frequency_highFreqRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.250,
    time_window_step=0.250,
    desired_frequencies=(0, 20),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.750, 0.250)
)
coherence_type = {
    'gamma_frequency_coherence_medFreqRes1': gamma_frequency_medFreqRes1,
    'gamma_frequency_coherence_medFreqRes2': gamma_frequency_medFreqRes2,
    'gamma_frequency_coherence_highTimeRes': gamma_frequency_highTimeRes,
    'gamma_frequency_coherence_highFreqRes': gamma_frequency_highFreqRes,
    'low_frequencies_coherence_highTimeRes': low_frequency_highTimeRes,
    'low_frequencies_coherence_highFreqRes': low_frequency_highFreqRes,
    'ripple_frequencies_coherence': ripple_frequency
}
ripple_covariates = ['is_spike', 'session_time',
                     'ripple_trajectory', 'ripple_direction']


def estimate_ripple_coherence(epoch_index):
    ripple_times = get_epoch_ripples(
        epoch_index, animals, sampling_frequency=sampling_frequency)

    # Compare before ripple to after ripple
    for coherence_name in coherence_type:
        ripple_triggered_coherence(
            epoch_index, animals, ripple_times,
            coherence_name=coherence_name,
            multitaper_params=coherence_type[coherence_name])
        ripple_triggered_canonical_coherence(
            epoch_index, animals, ripple_times,
            coherence_name=coherence_name,
            multitaper_params=coherence_type[coherence_name])

    # Compare different types of ripples
    ripple_info = decode_ripple(
        epoch_index, animals, ripple_times)[0]
    save_ripple_info(epoch_index, ripple_info)

    for covariate in ripple_covariates:
        for coherence_name in coherence_type:
            coherence_by_ripple_type(
                epoch_index, animals, ripple_info, covariate,
                coherence_name=coherence_name,
                multitaper_params=coherence_type[coherence_name])
            canonical_coherence_by_ripple_type(
                epoch_index, animals, ripple_info, covariate,
                coherence_name=coherence_name,
                multitaper_params=coherence_type[coherence_name])


def main():
    try:
        print('\n#############################################')
        print('Script start time: {}'.format(datetime.now()))
        print('#############################################\n')
        print('Git Hash:')
        print(run(['git', 'rev-parse', 'HEAD'],
                  stdout=PIPE, universal_newlines=True).stdout)
        epoch_index = (argv[1], int(argv[2]),
                       int(argv[3]))  # animal, day, epoch
        estimate_ripple_coherence(epoch_index)
        print('Script end time: {}'.format(datetime.now()))
    except IndexError:
        exit('Need three arguments to define epoch. '
             'Only gave {}.'.format(len(argv) - 1))

if __name__ == '__main__':
    exit(main())
