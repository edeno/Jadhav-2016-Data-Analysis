'''Exectue set of functions for each epoch
'''
from argparse import ArgumentParser
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
from sys import exit, stdout

from src.analysis import (canonical_coherence_by_ripple_type,
                          coherence_by_ripple_type,
                          decode_ripple_clusterless,
                          detect_epoch_ripples, group_delay_by_ripple_type,
                          ripple_triggered_canonical_coherence,
                          ripple_triggered_coherence,
                          ripple_triggered_group_delay)
from src.data_processing import (get_LFP_dataframe, make_tetrode_dataframe,
                                 make_tetrode_pair_info,
                                 save_multitaper_parameters,
                                 save_ripple_info,
                                 save_tetrode_info, save_tetrode_pair_info)
from src.parameters import (ANIMALS, SAMPLING_FREQUENCY,
                            MULTITAPER_PARAMETERS, FREQUENCY_BANDS,
                            RIPPLE_COVARIATES, ALPHA)


def estimate_ripple_coherence(epoch_key):
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, sampling_frequency=SAMPLING_FREQUENCY)

    tetrode_info = make_tetrode_dataframe(ANIMALS)[epoch_key]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    save_tetrode_info(epoch_key, tetrode_info)

    tetrode_pair_info = make_tetrode_pair_info(tetrode_info)
    save_tetrode_pair_info(epoch_key, tetrode_pair_info)

    lfps = {tetrode_key: get_LFP_dataframe(tetrode_key, ANIMALS)
            for tetrode_key in tetrode_info.index}

    # Compare before ripple to after ripple
    for parameters_name, parameters in MULTITAPER_PARAMETERS.items():
        ripple_triggered_coherence(
            lfps, ripple_times,
            multitaper_parameter_name=parameters_name,
            multitaper_params=parameters)
        ripple_triggered_canonical_coherence(
            lfps, epoch_key, tetrode_info, ripple_times,
            multitaper_parameter_name=parameters_name,
            multitaper_params=parameters)
        for frequency_band_name, frequency_band in FREQUENCY_BANDS.items():
            ripple_triggered_group_delay(
                tetrode_pair_info, parameters_name, frequency_band,
                frequency_band_name, alpha=ALPHA)
        save_multitaper_parameters(
            epoch_key, parameters_name, parameters)

    # Compare different types of ripples
    ripple_info = decode_ripple_clusterless(
        epoch_key, ANIMALS, ripple_times)[0]
    save_ripple_info(epoch_key, ripple_info)

    for covariate in RIPPLE_COVARIATES:
        for parameters_name, parameters in MULTITAPER_PARAMETERS.items():
            coherence_by_ripple_type(
                lfps, ripple_info, covariate,
                multitaper_parameter_name=parameters_name,
                multitaper_params=parameters)
            canonical_coherence_by_ripple_type(
                lfps, epoch_key, tetrode_info, ripple_info, covariate,
                multitaper_parameter_name=parameters_name,
                multitaper_params=parameters)
            for (frequency_band_name,
                 frequency_band) in FREQUENCY_BANDS.items():
                group_delay_by_ripple_type(
                    tetrode_pair_info, ripple_info, covariate,
                    parameters_name, frequency_band, frequency_band_name,
                    alpha=ALPHA)


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

    estimate_ripple_coherence(epoch_key)

    logger.info('Finished Processing')

if __name__ == '__main__':
    exit(main())
