'''Exectue set of functions for each epoch
'''
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import combinations
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
from sys import exit, stdout

from src.analysis import (decode_ripple_clusterless,
                          detect_epoch_ripples,
                          ripple_triggered_connectivity,
                          connectivity_by_ripple_type,
                          ripple_locked_firing_rate_change,
                          ripple_cross_correlation, ripple_spike_coherence,
                          compare_spike_coherence)
from src.data_processing import (get_LFP_dataframe, make_tetrode_dataframe,
                                 make_neuron_dataframe, save_xarray,
                                 get_interpolated_position_dataframe)
from src.parameters import (ANIMALS, SAMPLING_FREQUENCY,
                            MULTITAPER_PARAMETERS, FREQUENCY_BANDS,
                            REPLAY_COVARIATES)


def estimate_ripple_spike_connectivity(epoch_key, n_boot_samples=1000):
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, SAMPLING_FREQUENCY)
    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False).query('numspikes > 0')

    results = dict()

    results['firing_rate/all_ripples'] = ripple_locked_firing_rate_change(
        ripple_times.values, neuron_info, ANIMALS, SAMPLING_FREQUENCY,
        window_offset=(-0.100, 0.100), formula='bs(time, df=5)',
        n_boot_samples=n_boot_samples)
    results['cross_correlation/all_ripples'] = ripple_cross_correlation(
        ripple_times.values, neuron_info, ANIMALS, SAMPLING_FREQUENCY,
        window_offset=(-0.100, 0.100))
    coherence_all_ripples = ripple_spike_coherence(
        ripple_times, neuron_info, ANIMALS, SAMPLING_FREQUENCY,
        MULTITAPER_PARAMETERS['10Hz_Resolution'], (-0.100, 0.100))
    results['coherence/all_ripples'] = compare_spike_coherence(
        coherence_all_ripples.isel(time=0),
        coherence_all_ripples.isel(time=1), SAMPLING_FREQUENCY,
        'After Ripple - Before Ripple')

    # Compare different types of replay
    replay_info, state_probability, posterior_density = (
        decode_ripple_clusterless(epoch_key, ANIMALS, ripple_times))

    for covariate in REPLAY_COVARIATES:

        coherence = OrderedDict()

        for level_name, df in replay_info.groupby(covariate):
            level_ripple_times = df.loc[:, ['start_time', 'end_time']].values
            subgroup_name = '/'.join((covariate, level_name))
            results['firing_rate/' + subgroup_name] = (
                ripple_locked_firing_rate_change(
                    level_ripple_times, neuron_info, ANIMALS,
                    SAMPLING_FREQUENCY, window_offset=(-0.100, 0.100),
                    formula='bs(time, df=5)',
                    n_boot_samples=n_boot_samples))
            results['cross_correlation/' + subgroup_name] = (
                ripple_cross_correlation(
                    level_ripple_times, neuron_info, ANIMALS,
                    SAMPLING_FREQUENCY, window_offset=(-0.100, 0.100)))
            coherence[level_name] = ripple_spike_coherence(
                ripple_times, neuron_info, ANIMALS, SAMPLING_FREQUENCY,
                MULTITAPER_PARAMETERS['10Hz_Resolution'], (0.00, 0.100))

        for level1, level2 in combinations(coherence.keys(), 2):
            comparison_name = '-'.join((level2, level1))
            subgroup_name = '/'.join(('coherence', covariate, comparison_name))
            results[subgroup_name] = compare_spike_coherence(
                coherence[level2], coherence[level1], SAMPLING_FREQUENCY,
                comparison_name)

    results['replay_info'] = replay_info.reset_index().to_xarray()
    results['state_probability'] = state_probability
    results['posterior_density'] = posterior_density

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
    ripple_info = decode_ripple_clusterless(
        epoch_key, ANIMALS, ripple_times)[0]

    for covariate in REPLAY_COVARIATES:
        for parameters_name, parameters in MULTITAPER_PARAMETERS.items():
            connectivity_by_ripple_type(
                lfps, epoch_key, tetrode_info,
                ripple_info.query('ripple_state_probability >= 0.7'),
                covariate,
                parameters, FREQUENCY_BANDS,
                multitaper_parameter_name=parameters_name)

    save_xarray(
        epoch_key, ripple_info.reset_index().to_xarray(), '/ripple_info')


def decode_ripples(epoch_key):

    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, sampling_frequency=SAMPLING_FREQUENCY)

    # Compare different types of ripples
    replay_info, state_probability, posterior_density = (
        decode_ripple_clusterless(epoch_key, ANIMALS, ripple_times))

    position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)

    results = dict()
    results['replay_info'] = replay_info.reset_index().to_xarray()
    results['position_info'] = position_info.to_xarray()
    results['state_probability'] = state_probability
    results['posterior_density'] = posterior_density

    for group_name, data in results.items():
        save_xarray(epoch_key, data, group_name)


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

    estimate_ripple_spike_connectivity(epoch_key,
                                       window_offset=(-0.100, 0.100))

    logger.info('Finished Processing')

if __name__ == '__main__':
    exit(main())
