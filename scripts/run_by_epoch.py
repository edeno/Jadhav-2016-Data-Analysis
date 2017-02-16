'''Exectue set of functions for each epoch
'''
from argparse import ArgumentParser
from collections import namedtuple
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from subprocess import PIPE, run
from sys import exit, stdout

from dask import multiprocessing

from src.analysis import (canonical_coherence_by_ripple_type,
                          coherence_by_ripple_type,
                          decode_ripple_clusterless, detect_epoch_ripples,
                          ripple_triggered_canonical_coherence,
                          ripple_triggered_coherence, save_ripple_info)

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
    desired_frequencies=(12, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.450, 0.400)
)
gamma_frequency_medFreqRes1 = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.100,
    time_window_step=0.100,
    desired_frequencies=(12, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.500, 0.400)
)
gamma_frequency_medFreqRes2 = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.200,
    time_window_step=0.200,
    desired_frequencies=(12, 125),
    time_halfbandwidth_product=1,
    window_of_interest=(-0.600, 0.400)
)
gamma_frequency_highFreqRes = dict(
    sampling_frequency=sampling_frequency,
    time_window_duration=0.400,
    time_window_step=0.400,
    desired_frequencies=(12, 125),
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
    ripple_times = detect_epoch_ripples(
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
    ripple_info = decode_ripple_clusterless(
        epoch_index, animals, ripple_times,
        scheduler=multiprocessing.get,
        scheduler_kwargs=dict(num_workers=4))[0]
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


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=DEBUG,
        default=INFO,
    )
    return parser.parse_args()


def get_logger():
    formatter = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = StreamHandler(stream=stdout)
    handler.setFormatter(formatter)
    logger = getLogger()
    logger.addHandler(handler)
    return logger


def main():
    args = get_command_line_arguments()
    logger = get_logger()
    logger.setLevel(args.log_level)

    epoch_index = (args.Animal, args.Day, args.Epoch)
    logger.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_index))
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logger.info('Git Hash: {git_hash}'.format(git_hash=git_hash))

    estimate_ripple_coherence(epoch_index)

    logger.info('Finished Processing')

if __name__ == '__main__':
    exit(main())
