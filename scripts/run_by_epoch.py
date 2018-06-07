'''Exectue set of functions for each epoch
'''
import logging
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
from sys import exit

import numpy as np
import xarray as xr

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         get_LFP_dataframe,
                                         get_spike_indicator_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe, save_xarray)
from spectral_connectivity import Connectivity, Multitaper
from src.analysis import (_center_time, adjusted_coherence_magnitude,
                          connectivity_by_ripple_type,
                          decode_ripple_clusterless, detect_epoch_ripples,
                          get_hippocampal_theta, get_ripple_indicator,
                          get_ripple_locked_spikes,
                          ripple_triggered_connectivity)
from src.parameters import (ANIMALS, FREQUENCY_BANDS, MULTITAPER_PARAMETERS,
                            PROCESSED_DATA_DIR, REPLAY_COVARIATES,
                            SAMPLING_FREQUENCY)
from src.spike_models import (DROP_COLUMNS, fit_1D_position,
                              fit_1D_position_by_speed,
                              fit_1D_position_by_speed_and_task,
                              fit_1D_position_by_speed_by_task,
                              fit_1D_position_by_task, fit_2D_position,
                              fit_2D_position_by_speed,
                              fit_2D_position_by_speed_and_task,
                              fit_2D_position_by_task, fit_hippocampal_theta,
                              fit_hippocampal_theta_by_1D_position,
                              fit_position_constant, fit_replay,
                              fit_ripple_constant, fit_ripple_over_time,
                              fit_task, fit_task_by_turn, fit_turn)
from src.to_rasterVis import export_session_and_neuron_info
from src.replicate import swr_stats


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
                replay_info, covariate, parameters, FREQUENCY_BANDS,
                multitaper_parameter_name=parameters_name)

    save_xarray(
        epoch_key, replay_info.to_xarray(), '/replay_info')


def estimate_ripple_locked_spiking(epoch_key, ripple_times, replay_info,
                                   neuron_info, window_offset=(-0.500, 0.500)):

    ripple_locked_spikes = [get_ripple_locked_spikes(
        neuron_key, ripple_times, ANIMALS, SAMPLING_FREQUENCY,
        window_offset)
        for neuron_key in neuron_info.index]

    results = {}

    results['ripple/constant'] = xr.concat(
        [fit_ripple_constant(data, SAMPLING_FREQUENCY)
         for data in ripple_locked_spikes], dim=neuron_info.neuron_id)
    results['ripple/over_time'] = xr.concat(
        [fit_ripple_over_time(data, SAMPLING_FREQUENCY)
         for data in ripple_locked_spikes], dim=neuron_info.neuron_id)
    results['ripple/replay_state'] = xr.concat(
        [fit_replay(data, SAMPLING_FREQUENCY, replay_info, 'predicted_state')
         for data in ripple_locked_spikes], dim=neuron_info.neuron_id)
    results['ripple/session_time'] = xr.concat(
        [fit_replay(data, SAMPLING_FREQUENCY, replay_info, 'session_time')
         for data in ripple_locked_spikes], dim=neuron_info.neuron_id)

    for group_name, data in results.items():
        save_xarray(PROCESSED_DATA_DIR, epoch_key, data, group_name)


def get_replay(epoch_key):
    ripple_times, replay_info = export_session_and_neuron_info(
        epoch_key, PROCESSED_DATA_DIR)
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                replay_info.to_xarray(), '/replay_info')
    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                ripple_times.to_xarray(), '/ripple_times')


def estimate_spike_task_1D_information(
        epoch_key, ripple_indicator, neuron_info, position_info):

    constant_model = []
    task_model = []
    turn_model = []
    task_by_turn_model = []
    position_model = []
    position_by_task_model = []
    position_by_speed_model = []
    position_by_speed_and_task_model = []
    position_by_speed_by_task_model = []

    for neuron_key in neuron_info.index:
        spikes = get_spike_indicator_dataframe(
            neuron_key, ANIMALS).rename('is_spike')
        non_ripple_data = (
            position_info.join(spikes)
            .loc[~ripple_indicator & position_info.is_correct]
            .drop(DROP_COLUMNS, axis=1)
            .dropna())

        constant_model.append(
            fit_position_constant(non_ripple_data, SAMPLING_FREQUENCY))
        task_model.append(
            fit_task(non_ripple_data, SAMPLING_FREQUENCY))
        turn_model.append(
            fit_turn(non_ripple_data, SAMPLING_FREQUENCY))
        task_by_turn_model.append(
            fit_task_by_turn(non_ripple_data, SAMPLING_FREQUENCY))
        position_model.append(
            fit_1D_position(non_ripple_data, SAMPLING_FREQUENCY))
        position_by_task_model.append(
            fit_1D_position_by_task(non_ripple_data, SAMPLING_FREQUENCY))
        position_by_speed_model.append(
            fit_1D_position_by_speed(non_ripple_data, SAMPLING_FREQUENCY))
        position_by_speed_and_task_model.append(
            fit_1D_position_by_speed_and_task(
                non_ripple_data, SAMPLING_FREQUENCY))
        position_by_speed_by_task_model.append(
            fit_1D_position_by_speed_by_task(
                non_ripple_data, SAMPLING_FREQUENCY))

    results = {}

    results['non_ripple/constant_model'] = xr.concat(
        constant_model, dim=neuron_info.neuron_id)
    results['non_ripple/task_model'] = xr.concat(
        task_model, dim=neuron_info.neuron_id)
    results['non_ripple/turn_model'] = xr.concat(
        turn_model, dim=neuron_info.neuron_id)
    results['non_ripple/task_by_turn_model'] = xr.concat(
        task_by_turn_model, dim=neuron_info.neuron_id)
    results['non_ripple/1D_position_model'] = xr.concat(
        position_model, dim=neuron_info.neuron_id)
    results['non_ripple/1D_position_by_task_model'] = xr.concat(
        position_by_task_model, dim=neuron_info.neuron_id)
    results['non_ripple/1D_position_by_speed_model'] = xr.concat(
        position_by_speed_model, dim=neuron_info.neuron_id)
    results['non_ripple/1D_position_by_speed_and_task_model'] = xr.concat(
        position_by_speed_and_task_model, dim=neuron_info.neuron_id)
    results['non_ripple/1D_position_by_speed_by_task_model'] = xr.concat(
        position_by_speed_by_task_model, dim=neuron_info.neuron_id)

    for group_name, data in results.items():
        save_xarray(PROCESSED_DATA_DIR, epoch_key, data, group_name)


def estimate_theta_spike_field_coherence(epoch_key, neuron_info,
                                         position_info):
    theta = get_hippocampal_theta(epoch_key, ANIMALS, SAMPLING_FREQUENCY)

    theta_phase_model = []
    theta_phase_by_position_model = []

    for neuron_key in neuron_info.index:
        spikes = get_spike_indicator_dataframe(
            neuron_key, ANIMALS).rename('is_spike')
        data = (theta.join(spikes)
                     .join(position_info)
                     .loc[position_info.speed >= 4 & position_info.is_correct]
                .dropna())
        theta_phase_model.append(
            fit_hippocampal_theta(data, SAMPLING_FREQUENCY))
        theta_phase_by_position_model.append(
            fit_hippocampal_theta_by_1D_position(data, SAMPLING_FREQUENCY))

    results = {}

    results['non_ripple/theta_phase_model'] = xr.concat(
        theta_phase_model, dim=neuron_info.neuron_id)
    results['non_ripple/theta_by_1D_position_model'] = xr.concat(
        theta_phase_by_position_model, dim=neuron_info.neuron_id)
    for group_name, data in results.items():
        save_xarray(PROCESSED_DATA_DIR, epoch_key, data, group_name)


def estimate_2D_spike_task_information(
        epoch_key, ripple_indicator, neuron_info, position_info):

    position_2D_model = []
    position_2D_by_task_model = []
    position_2D_by_speed_model = []
    position_2D_by_speed_and_task_model = []

    for neuron_key in neuron_info.index:
        spikes = get_spike_indicator_dataframe(
            neuron_key, ANIMALS).rename('is_spike')
        non_ripple_data = (
            position_info.join(spikes)
            .loc[~ripple_indicator
                 & position_info.is_correct.fillna(True)]
            .drop(DROP_COLUMNS, axis=1)
            .dropna())

        position_2D_model.append(fit_2D_position(
            non_ripple_data, neuron_key, ANIMALS, SAMPLING_FREQUENCY))
        position_2D_by_task_model.append(fit_2D_position_by_task(
            non_ripple_data, neuron_key, ANIMALS, SAMPLING_FREQUENCY))
        position_2D_by_speed_model.append(fit_2D_position_by_speed(
            non_ripple_data, neuron_key, ANIMALS, SAMPLING_FREQUENCY))
        position_2D_by_speed_and_task_model.append(
            fit_2D_position_by_speed_and_task(
                non_ripple_data, neuron_key, ANIMALS, SAMPLING_FREQUENCY))

    results = {}
    results['non_ripple/2D_position_model'] = xr.concat(
        position_2D_model, dim=neuron_info.neuron_id)
    results['non_ripple/2D_position_by_task_model'] = xr.concat(
        position_2D_by_task_model, dim=neuron_info.neuron_id)
    results['non_ripple/2D_position_by_speed_model'] = xr.concat(
        position_2D_by_speed_model, dim=neuron_info.neuron_id)
    results['non_ripple/2D_position_by_speed_and_task_model'] = xr.concat(
        position_2D_by_speed_and_task_model, dim=neuron_info.neuron_id)

    for group_name, data in results.items():
        save_xarray(PROCESSED_DATA_DIR, epoch_key, data, group_name)


def estimate_ripple_locked_spike_spike_coherence(
        epoch_key, ripple_times, neuron_info, window_offset=(-0.250, 0.250)):

    spikes = np.stack(
        [get_ripple_locked_spikes(
            key, ripple_times, ANIMALS, SAMPLING_FREQUENCY, window_offset
        ).unstack(0).values for key in neuron_info.index], axis=-1)

    m = Multitaper(spikes, SAMPLING_FREQUENCY,
                   time_window_duration=window_offset[1],
                   time_halfbandwidth_product=1)
    c = Connectivity.from_multitaper(m)

    dims = ['time', 'frequency', 'neuron1', 'neuron2']
    coherence_magnitude = adjusted_coherence_magnitude(
        spikes, SAMPLING_FREQUENCY, m, c)
    coherence_difference = np.squeeze(np.diff(coherence_magnitude, axis=0))

    data_vars = {
        'coherence_magnitude': (dims, coherence_magnitude),
        'coherence_difference': (dims[1:], coherence_difference)
    }
    coords = {
        'time': _center_time(c.time),
        'frequency': c.frequencies,
        'neuron1': neuron_info.neuron_id.values,
        'neuron2': neuron_info.neuron_id.values,
    }
    data = xr.Dataset(data_vars, coords=coords)

    save_xarray(PROCESSED_DATA_DIR, epoch_key, data,
                'ripple/spike_spike_coherence')


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
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)

    def _signal_handler(signal_code, frame):
        logging.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    # position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)
    # ripple_times = detect_epoch_ripples(
    #     epoch_key, ANIMALS, SAMPLING_FREQUENCY, position_info)
    # ripple_indicator = get_ripple_indicator(
    #     epoch_key, ANIMALS, ripple_times)
    # neuron_info = make_neuron_dataframe(ANIMALS).xs(
    #     epoch_key, drop_level=False).query('numspikes > 0')
    #
    # position_info['task_by_turn'] = (
    #     position_info.task + '_' + position_info.turn)
    # replay_info, _, _ = decode_ripple_clusterless(
    #     epoch_key, ANIMALS, ripple_times, position_info
    #
    # logging.info('Estimating ripple-locked spiking models...')
    # estimate_ripple_locked_spiking(
    #     epoch_key, ripple_times, replay_info, neuron_info)
    # logging.info('Estimating theta spike-field coherence...')
    # estimate_theta_spike_field_coherence(
    #     epoch_key, neuron_info, position_info)
    # logging.info('Estimating ripple-locked spike-spike coherence...')
    # estimate_ripple_locked_spike_spike_coherence(
    #     epoch_key, ripple_times, neuron_info)
    # logging.info('Estimating non-ripple 1D spike models...')
    # estimate_spike_task_1D_information(
    #     epoch_key, ripple_indicator, neuron_info, position_info)
    # logging.info('Estimating non-ripple 2D spike models...')
    # estimate_2D_spike_task_information(
    #     epoch_key, ripple_indicator, neuron_info, position_info)
    # get_replay(epoch_key)
    stats = (swr_stats(epoch_key, ANIMALS, SAMPLING_FREQUENCY)
             .reset_index().to_xarray())
    save_xarray(PROCESSED_DATA_DIR, epoch_key, stats, '/replicate/swr_stats')
    logging.info('Finished Processing')


if __name__ == '__main__':
    exit(main())
