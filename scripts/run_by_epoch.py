'''Exectue set of functions for each epoch
'''
import logging
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run
from sys import exit

import pandas as pd
import xarray as xr

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         get_LFPs,
                                         get_spike_indicator_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe, save_xarray)
from spectral_connectivity import Connectivity, Multitaper
from src.analysis import (_center_time, adjusted_coherence_magnitude,
                          connectivity_by_ripple_type,
                          decode_ripple_clusterless, detect_epoch_ripples,
                          get_hippocampal_theta, get_ripple_locked_spikes,
                          ripple_triggered_connectivity)
from src.parameters import (ANIMALS, FREQUENCY_BANDS, MULTITAPER_PARAMETERS,
                            PROCESSED_DATA_DIR, REPLAY_COVARIATES,
                            SAMPLING_FREQUENCY)
from src.replicate import swr_stats
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
                              fit_ripple_over_time_with_other_neurons,
                              fit_task, fit_task_by_turn, fit_turn)
from src.to_rasterVis import export_session_and_neuron_info


def estimate_lfp_ripple_connectivity(epoch_key, ripple_times, replay_info):

    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)
        & (tetrode_info.numcells > 0)
        & tetrode_info.area.isin(['PFC', 'CA1', 'iCA1'])
    ]

    lfps = get_LFPs(tetrode_info.index, ANIMALS)

    for parameters_name, parameters in MULTITAPER_PARAMETERS.items():
        # Compare all ripples
        ripple_triggered_connectivity(
            lfps, epoch_key, tetrode_info, ripple_times, parameters,
            SAMPLING_FREQUENCY, FREQUENCY_BANDS,
            multitaper_parameter_name=parameters_name)

    # Compare different types of ripples
    for covariate in REPLAY_COVARIATES:
        for parameters_name, parameters in MULTITAPER_PARAMETERS.items():
            connectivity_by_ripple_type(
                lfps, epoch_key, tetrode_info,
                replay_info, covariate, parameters, SAMPLING_FREQUENCY,
                FREQUENCY_BANDS, multitaper_parameter_name=parameters_name)


def estimate_ripple_locked_spiking(epoch_key, ripple_times, replay_info,
                                   neuron_info, window_offset=(-0.500, 0.500)):

    ripple_locked_spikes = get_ripple_locked_spikes(
        neuron_info.index, ripple_times, ANIMALS, SAMPLING_FREQUENCY)
    ripple_locked_spikes['time'] = (
        ripple_locked_spikes.index.get_level_values('time').total_seconds())
    ripple_locked_spikes['ripple_number'] = (
        ripple_locked_spikes.index.get_level_values('ripple_number'))
    ripple_locked_spikes = pd.merge(
        ripple_locked_spikes, replay_info, on='ripple_number')
    results = {}

    results['all_ripples/constant'] = xr.concat(
        [fit_ripple_constant(
            neuron_key, ripple_locked_spikes, SAMPLING_FREQUENCY, neuron_info)
         for neuron_key in neuron_info.index], dim=neuron_info.neuron_id)
    results['all_ripples/over_time'] = xr.concat(
        [fit_ripple_over_time(
            neuron_key, ripple_locked_spikes, SAMPLING_FREQUENCY, neuron_info)
         for neuron_key in neuron_info.index], dim=neuron_info.neuron_id)
    results['all_ripples/replay_state'] = xr.concat(
        [fit_replay(
            neuron_key, ripple_locked_spikes, SAMPLING_FREQUENCY,
            neuron_info, 'predicted_state')
         for neuron_key in neuron_info.index], dim=neuron_info.neuron_id)
    results['all_ripples/over_time_and_auto'] = xr.concat(
        [fit_ripple_over_time_with_other_neurons(
            neuron_key, ripple_locked_spikes, SAMPLING_FREQUENCY,
            neuron_info, [])
         for neuron_key in neuron_info.index], dim=neuron_info.neuron_id)
    results['all_ripples/over_time_CA1'] = xr.concat(
        [fit_ripple_over_time_with_other_neurons(
            neuron_key, ripple_locked_spikes, SAMPLING_FREQUENCY,
            neuron_info, ['iCA1'])
         for neuron_key in neuron_info.index], dim=neuron_info.neuron_id)
    results['all_ripples/over_time_PFC'] = xr.concat(
        [fit_ripple_over_time_with_other_neurons(
            neuron_key, ripple_locked_spikes, SAMPLING_FREQUENCY,
            neuron_info, ['PFC'])
         for neuron_key in neuron_info.index], dim=neuron_info.neuron_id)
    results['all_ripples/over_time_iCA1'] = xr.concat(
        [fit_ripple_over_time_with_other_neurons(
            neuron_key, ripple_locked_spikes, SAMPLING_FREQUENCY,
            neuron_info, ['iCA1'])
         for neuron_key in neuron_info.index], dim=neuron_info.neuron_id)
    for group_name, data in results.items():
        save_xarray(PROCESSED_DATA_DIR, epoch_key, data, group_name)


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
        epoch_key, ripple_times, neuron_info):

    ripple_locked_spikes = get_ripple_locked_spikes(
        neuron_info.index, ripple_times, ANIMALS, SAMPLING_FREQUENCY)
    ripple_locked_spikes = (ripple_locked_spikes.to_xarray().to_array()
                            .rename({'variable': 'neurons'})
                            .transpose('time', 'ripple_number', 'neurons')
                            .dropna('ripple_number'))

    m = Multitaper(ripple_locked_spikes.values,
                   sampling_frequency=SAMPLING_FREQUENCY,
                   time_window_duration=0.250,
                   time_window_step=0.250,
                   time_halfbandwidth_product=3,
                   start_time=-0.5)
    c = Connectivity.from_multitaper(m)

    dims = ['time', 'frequency', 'neuron1', 'neuron2']
    coherence_magnitude = adjusted_coherence_magnitude(
        ripple_locked_spikes.values, SAMPLING_FREQUENCY, m, c)
    coherence_difference = coherence_magnitude - coherence_magnitude[0, :]

    data_vars = {
        'coherence_magnitude': (dims, coherence_magnitude),
        'coherence_difference': (dims, coherence_difference)
    }
    coords = {
        'time': _center_time(c.time),
        'frequency': c.frequencies,
        'neuron1': neuron_info.neuron_id.values,
        'neuron2': neuron_info.neuron_id.values,
    }
    data = xr.Dataset(data_vars, coords=coords)

    save_xarray(PROCESSED_DATA_DIR, epoch_key, data,
                'all_ripples/spike_spike_coherence')


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

    position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)
    ripple_times = detect_epoch_ripples(
        epoch_key, ANIMALS, SAMPLING_FREQUENCY, position_info)
    replay_info, _, _ = decode_ripple_clusterless(
        epoch_key, ANIMALS, ripple_times, position_info=position_info)
    save_xarray(PROCESSED_DATA_DIR, epoch_key, replay_info.to_xarray(),
                '/replay_info')

    logging.info('Estimating ripple-locked LFP connectivity...')
    estimate_lfp_ripple_connectivity(epoch_key, ripple_times, replay_info)

    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    is_PFC_FS_interneuron = (
        (neuron_info.meanrate > 17.0) & (neuron_info.spikewidth < 0.3))
    is_CA1_FS_interneuron = (
        (neuron_info.meanrate > 7.0) & (neuron_info.spikewidth < 0.3))
    neuron_info = neuron_info.loc[
        (neuron_info.area.isin(['PFC']) & ~is_PFC_FS_interneuron
         & (neuron_info.numspikes > 0)) |
        (neuron_info.area.isin(['CA1', 'iCA1']) & ~is_CA1_FS_interneuron
         & (neuron_info.numspikes > 0))
    ]
    logging.info('Estimating ripple-locked spike-spike coherence...')
    estimate_ripple_locked_spike_spike_coherence(
        epoch_key, ripple_times, neuron_info)

    logging.info('Estimating ripple-locked spiking models...')
    estimate_ripple_locked_spiking(
        epoch_key, ripple_times, replay_info, neuron_info)

    logging.info('Replicating Jadhav 2016 analysis...')
    stats = (swr_stats(epoch_key, ANIMALS, SAMPLING_FREQUENCY)
             .reset_index().to_xarray())
    save_xarray(PROCESSED_DATA_DIR, epoch_key, stats, '/replicate/swr_stats')

    logging.info('Exporting data to rasterVis...')
    export_session_and_neuron_info(
        epoch_key, PROCESSED_DATA_DIR, ripple_times, replay_info)

    logging.info('Finished Processing')


if __name__ == '__main__':
    exit(main())
