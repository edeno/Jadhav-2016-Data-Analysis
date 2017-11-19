'''Exectue set of functions for each epoch
'''
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import combinations
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
from sys import exit, stdout

from loren_frank_data_processing import (get_LFP_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe, save_xarray)
from src.analysis import (compare_spike_coherence, connectivity_by_ripple_type,
                          decode_ripple_clusterless, detect_epoch_ripples,
                          ripple_locked_firing_rate_change,
                          ripple_spike_coherence,
                          ripple_triggered_connectivity)
from src.parameters import (ANIMALS, FREQUENCY_BANDS, MULTITAPER_PARAMETERS,
                            REPLAY_COVARIATES, SAMPLING_FREQUENCY)


def estimate_ripple_spike_connectivity(epoch_key, n_boot_samples=1000):
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, SAMPLING_FREQUENCY)
    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False).query('numspikes > 0')

    results = dict()

    results['firing_rate/all_ripples/bs_time'] = ripple_locked_firing_rate_change(
        ripple_times, neuron_info, ANIMALS, SAMPLING_FREQUENCY,
        window_offset=(-0.100, 0.100), formula='bs(time, df=5)',
        n_boot_samples=n_boot_samples)

    for group_name, data in results.items():
        save_xarray(epoch_key, data, group_name)


def estimate_ripple_coherence(epoch_key):
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, sampling_frequency=SAMPLING_FREQUENCY)

    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]

    lfps = {tetrode_key: get_LFP_dataframe(tetrode_key, ANIMALS)
            for tetrode_key in tetrode_info.index}

    for parameters_name, parameters in MULTITAPER_PARAMETERS.items():
        # Compare all ripples
        ripple_triggered_connectivity(
            lfps, epoch_key, tetrode_info, ripple_times, parameters,
            FREQUENCY_BANDS, multitaper_parameter_name=parameters_name)

    # Compare different types of ripples
    replay_info = decode_ripple_clusterless(
        epoch_key, ANIMALS, ripple_times)[0]

    for covariate in REPLAY_COVARIATES:
        for parameters_name, parameters in MULTITAPER_PARAMETERS.items():
            connectivity_by_ripple_type(
                lfps, epoch_key, tetrode_info,
                replay_info.query('ripple_state_probability >= 0.8'),
                covariate,
                parameters, FREQUENCY_BANDS,
                multitaper_parameter_name=parameters_name)

    save_xarray(
        epoch_key, replay_info.to_xarray(), '/replay_info')


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

    def _signal_handler(signal_code, frame):
        logger.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logger.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logger.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    estimate_ripple_spike_connectivity(epoch_key)

    logger.info('Finished Processing')


if __name__ == '__main__':
    exit(main())
