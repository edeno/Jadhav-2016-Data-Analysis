#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import collections
sys.path.append(os.path.join(os.path.abspath(os.path.pardir), 'src'))
import ripple_detection
import ripple_decoding
import analysis

sampling_frequency = 1500
Animal = collections.namedtuple('Animal', {'directory', 'short_name'})
animals = {
    'HPa': Animal(directory='HPa_direct', short_name='HPa'),
    'HPc': Animal(directory='HPc_direct', short_name='HPc')
}
gamma_frequency_params = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.600,
    time_window_step=0.600,
    desired_frequencies=[20, 100],
    time_halfbandwidth_product=3,
    window_of_interest=(-2.100, 2.100)
)
low_frequency_params = dict(
    sampling_frequency=1500,
    time_window_duration=3.000,
    time_window_step=3.000,
    desired_frequencies=[3, 20],
    time_halfbandwidth_product=3,
    window_of_interest=(-7.500, 7.500)
)
coherence_type = {
    'gamma_coherence': gamma_frequency_params,
    'low_frequency_coherence': low_frequency_params
}
ripple_covariates = ['is_spike', 'session_time', 'ripple_trajectory', 'ripple_direction']


def coherence_by_ripple_type(epoch_index):
    ripple_times = ripple_detection.get_epoch_ripples(
        epoch_index, animals, sampling_frequency=sampling_frequency)
    ripple_info = ripple_decoding.decode_ripple(
        epoch_index, animals, ripple_times)[0]
    analysis.save_ripple_info(epoch_index, ripple_info)
    for covariate in ripple_covariates:
        for coherence_name in coherence_type:
            analysis.coherence_by_ripple_type(
                epoch_index, animals, ripple_info, covariate,
                coherence_name=coherence_name,
                multitaper_params=coherence_type[coherence_name])


def main():
    try:
        epoch_index = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))  # animal, day, epoch
        coherence_by_ripple_type(epoch_index)
        print(os.getcwd())
    except IndexError:
        sys.exit('Need three arguments to define epoch. '
                 'Only gave {}.'.format(len(sys.argv)-1))

if __name__ == '__main__':
    sys.exit(main())
