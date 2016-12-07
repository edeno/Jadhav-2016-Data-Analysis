#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import collections
import datetime
sys.path.append(os.path.join(os.path.abspath(os.path.pardir), 'src'))
import ripple_detection
import ripple_decoding
import analysis

sampling_frequency = 1500
Animal = collections.namedtuple('Animal', {'directory', 'short_name'})
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
    window_of_interest=(-0.200, 0.400)
)
gamma_frequency_highTimeRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.050,
    time_window_step=0.050,
    desired_frequencies=(20, 100),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.200, 0.400)
)
gamma_frequency_highFreqRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.100,
    time_window_step=0.100,
    desired_frequencies=(20, 100),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.200, 0.400)
)
low_frequency_highTimeRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.100,
    time_window_step=0.100,
    desired_frequencies=(0, 20),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.200, 0.400)
)
low_frequency_highFreqRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.500,
    time_window_step=0.500,
    desired_frequencies=(0, 20),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.500, 0.500)
)
coherence_type = {
    'gamma_frequency_coherence_highTimeRes': gamma_frequency_highTimeRes,
    'gamma_frequency_coherence_highFreqRes': gamma_frequency_highFreqRes,
    'low_frequencies_coherence_highTimeRes': low_frequency_highTimeRes,
    'low_frequencies_coherence_highFreqRes': low_frequency_highFreqRes,
    'ripple_frequencies_coherence': ripple_frequency
}
ripple_covariates = ['is_spike', 'session_time',
                     'ripple_trajectory', 'ripple_direction']


def coherence_by_ripple_type(epoch_index):
    ripple_times = ripple_detection.get_epoch_ripples(
        epoch_index, animals, sampling_frequency=sampling_frequency)

    # Compare before ripple to after ripple
    for coherence_name in coherence_type:
        analysis.ripple_triggered_coherence(
            epoch_index, animals, ripple_times,
            coherence_name=coherence_name,
            multitaper_params=coherence_type[coherence_name])
        analysis.ripple_triggered_canonical_coherence(
            epoch_index, animals, ripple_times,
            coherence_name=coherence_name,
            multitaper_params=coherence_type[coherence_name])

    # Compare different types of ripples
    ripple_info = ripple_decoding.decode_ripple(
        epoch_index, animals, ripple_times)[0]
    analysis.save_ripple_info(epoch_index, ripple_info)

    for covariate in ripple_covariates:
        for coherence_name in coherence_type:
            analysis.coherence_by_ripple_type(
                epoch_index, animals, ripple_info, covariate,
                coherence_name=coherence_name,
                multitaper_params=coherence_type[coherence_name])
            analysis.canonical_coherence_by_ripple_type(
                epoch_index, animals, ripple_info, covariate,
                coherence_name=coherence_name,
                multitaper_params=coherence_type[coherence_name])


def main():
    try:
        print('\n#############################################')
        print('Script start time: {}'.format(datetime.datetime.now()))
        print('#############################################\n')
        epoch_index = (sys.argv[1], int(sys.argv[2]),
                       int(sys.argv[3]))  # animal, day, epoch
        coherence_by_ripple_type(epoch_index)
        print('Script end time: {}'.format(datetime.datetime.now()))
    except IndexError:
        sys.exit('Need three arguments to define epoch. '
                 'Only gave {}.'.format(len(sys.argv) - 1))

if __name__ == '__main__':
    sys.exit(main())
