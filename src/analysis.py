'''Higher level functions for analyzing the data

'''
from copy import deepcopy
from itertools import product
from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import filtfilt, hilbert, remez
from scipy.stats import linregress

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         get_LFPs,
                                         get_multiunit_indicator_dataframe,
                                         get_spike_indicator_dataframe,
                                         get_trial_time,
                                         make_tetrode_dataframe,
                                         reshape_to_segments, save_xarray)
from loren_frank_data_processing.neurons import get_all_spike_indicators
from replay_classification import ClusterlessDecoder, SortedSpikeDecoder
from ripple_detection import Kay_ripple_detector
from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.statistics import (coherence_bias,
                                              coherence_rate_adjustment,
                                              fisher_z_transform,
                                              get_normal_distribution_p_values)
from spectral_connectivity.transforms import _sliding_window
from src.parameters import PROCESSED_DATA_DIR

from .spike_train import cross_correlate

logger = getLogger(__name__)

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max',
          'channel_4_max']
_BRAIN_AREAS = 'CA1'


def entire_session_connectivity(
    lfps, epoch_key, tetrode_info, multitaper_params,
        FREQUENCY_BANDS, multitaper_parameter_name='',
        group_name='entire_epoch', time_window_duration=2.0):
    lfps = pd.Panel(lfps)
    params = deepcopy(multitaper_params)
    params.pop('window_of_interest')
    m = Multitaper(
        lfps.values.squeeze().T,
        **params,
        start_time=lfps.major_axis.min())
    c = Connectivity.from_multitaper(m)
    save_power(
        c, tetrode_info, epoch_key,
        multitaper_parameter_name, group_name)
    save_coherence(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name)
    save_group_delay(
        c, m, FREQUENCY_BANDS, tetrode_info, epoch_key,
        multitaper_parameter_name, group_name)
    save_pairwise_spectral_granger(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name)
    save_canonical_coherence(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name)


def ripple_cross_correlation(ripple_times, neuron_info, animals,
                             sampling_frequency, window_offset=None):

    before_ripple, after_ripple = [], []

    for neuron_key in neuron_info.index:
        spikes = get_spike_indicator_dataframe(neuron_key, animals)
        ripple_locked_spikes = reshape_to_segments(
            spikes, ripple_times, window_offset, sampling_frequency
        ).unstack(level=0)
        before_ripple.append(ripple_locked_spikes.loc[:0].values)
        after_ripple.append(ripple_locked_spikes.loc[0:].values)

    correlation_before_ripple = correlate_neurons(
        before_ripple, neuron_info.neuron_id, sampling_frequency)
    correlation_after_ripple = correlate_neurons(
        after_ripple, neuron_info.neuron_id, sampling_frequency)
    time = pd.Index([min(window_offset), 0.0], name='time')
    return xr.concat(
        (correlation_before_ripple, correlation_after_ripple), dim=time)


def correlate_neurons(neurons, neuron_id, sampling_frequency):
    index = pd.MultiIndex.from_product((neuron_id, neuron_id),
                                       names=['neuron1', 'neuron2'])
    return xr.concat(
        [cross_correlate(neuron1, neuron2, sampling_frequency).to_xarray()
         for neuron1, neuron2 in product(neurons, neurons)],
        dim=index).unstack('concat_dim')


def ripple_spike_coherence(ripple_times, neuron_info, animals,
                           sampling_frequency, multitaper_parameters,
                           window_offset=(-0.100, 0.100)):
    ripple_locked_spikes = []

    for neuron_key in neuron_info.index:
        spikes = get_spike_indicator_dataframe(neuron_key, animals)
        ripple_locked_spikes.append(
            reshape_to_segments(
                spikes, ripple_times, window_offset, sampling_frequency
            ).unstack(level=0))
    m = Multitaper(
        np.stack(ripple_locked_spikes, axis=-1),
        **multitaper_parameters,
        start_time=ripple_locked_spikes[0].index.total_seconds().values[0])
    c = Connectivity.from_multitaper(m)
    n_trials = len(ripple_times)

    average_firing_rate = np.stack([np.mean(_sliding_window(
        neuron.values, m.n_time_samples_per_window, m.n_time_samples_per_step,
        axis=0), axis=(1, 2)) * sampling_frequency
        for neuron in ripple_locked_spikes], axis=1)

    coords = {
        'time': c.time,
        'frequency': c.frequencies,
        'neuron1': neuron_info.neuron_id.values,
        'neuron2': neuron_info.neuron_id.values
    }
    attrs = {
        'n_trials': n_trials,
        'n_tapers': m.n_tapers,
        'frequency_resolution': m.frequency_resolution
    }
    data_vars = {
        'average_firing_rate': (['time', 'neuron1'], average_firing_rate),
        'power': (['time', 'frequency', 'neuron1'], c.power()),
        'coherency': (['time', 'frequency', 'neuron1', 'neuron2'],
                      c.coherency()),
        'coherence_magnitude': (['time', 'frequency', 'neuron1', 'neuron2'],
                                c.coherence_magnitude()),
    }

    return xr.Dataset(data_vars, coords, attrs)


def compare_spike_coherence(condition1, condition2, sampling_frequency,
                            comparison_name='spike_coherence'):
    dt = 1.0 / sampling_frequency
    adjustment = coherence_rate_adjustment(
        condition1.average_firing_rate, condition2.average_firing_rate,
        condition1.power, dt=dt)

    adjusted_coherency1 = (adjustment.rename({'neuron1': 'neuron2'}) *
                           adjustment * condition1.coherency).transpose(
        'frequency', 'neuron1', 'neuron2')

    coherence_difference = (np.abs(condition2.coherency) -
                            np.abs(adjusted_coherency1))
    bias1 = coherence_bias(condition1.n_trials * condition1.n_tapers)
    bias2 = coherence_bias(condition2.n_trials * condition2.n_tapers)
    coherence_z_difference = fisher_z_transform(
        condition2.coherency.values, bias2, adjusted_coherency1.values, bias1)
    p_values = get_normal_distribution_p_values(coherence_z_difference)

    DIMS = ['frequency', 'neuron1', 'neuron2']

    data_vars = {
        'coherence_difference': (DIMS, coherence_difference),
        'coherence_z_difference': (DIMS, coherence_z_difference),
        'p_values': (DIMS, p_values),
    }
    coords = {
        'frequency': adjusted_coherency1.frequency,
        'neuron1': adjusted_coherency1.neuron1,
        'neuron2': adjusted_coherency1.neuron2
    }
    attrs = {
        'n_trials1': condition1.n_trials,
        'n_trials2': condition1.n_trials,
        'n_tapers': condition1.n_tapers,
        'comparison': comparison_name,
    }

    return xr.Dataset(data_vars, coords, attrs)


def connectivity_by_ripple_type(
    lfps, epoch_key, tetrode_info, ripple_info, ripple_covariate,
        multitaper_params, FREQUENCY_BANDS, multitaper_parameter_name=''):
    '''Computes the coherence at each level of a ripple covariate
    from the ripple info dataframe and the differences between those
    levels'''

    ripples_by_covariate = ripple_info.groupby(ripple_covariate)

    logger.info(
        'Computing for each level of the covariate "{covariate}":'.format(
            covariate=ripple_covariate))
    for level_name, ripples_df in ripples_by_covariate:
        ripple_times = _get_ripple_times(ripples_df)
        logger.info(
            '...Level: {level_name} ({num_ripples} ripples)'.format(
                level_name=level_name,
                num_ripples=len(ripple_times)))
        ripple_triggered_connectivity(
            lfps, epoch_key, tetrode_info, ripple_times, multitaper_params,
            FREQUENCY_BANDS,
            multitaper_parameter_name=multitaper_parameter_name,
            group_name='{covariate_name}/{level_name}'.format(
                covariate_name=ripple_covariate,
                level_name=level_name))


def ripple_triggered_connectivity(
    lfps, epoch_key, tetrode_info, ripple_times, multitaper_params,
        frequency_bands, multitaper_parameter_name='',
        group_name='all_ripples'):
    n_lfps = lfps.shape[1]
    n_pairs = int(n_lfps * (n_lfps - 1) / 2)
    params = deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')

    logger.info('Computing ripple-triggered {multitaper_parameter_name} '
                'for {num_pairs} pairs of electrodes'.format(
                    multitaper_parameter_name=multitaper_parameter_name,
                    num_pairs=n_pairs))

    ripple_locked_lfps = reshape_to_segments(
        lfps, ripple_times, window_offset=window_of_interest,
        sampling_frequency=params['sampling_frequency'])
    ripple_locked_lfps = (ripple_locked_lfps.to_xarray().to_array()
                          .rename({'variable': 'tetrodes'})
                          .transpose('time', 'ripple_number', 'tetrodes')
                          .dropna('ripple_number'))
    ripple_ERP = ripple_locked_lfps.mean('ripple_number').to_dataset('ERP')
    ripple_locked_lfps = (ripple_locked_lfps
                          - ripple_locked_lfps.mean(['ripple_number']))
    start_time = ripple_locked_lfps.time.min().values / np.timedelta64(1, 's')

    m = Multitaper(
        ripple_locked_lfps.values,
        **params,
        start_time=start_time)
    c = Connectivity.from_multitaper(m)

    save_ERP(epoch_key, ripple_ERP, multitaper_parameter_name, group_name)

    save_power(
        c, tetrode_info, epoch_key,
        multitaper_parameter_name, group_name)
    save_coherence(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name)
    save_group_delay(
        c, m, frequency_bands, tetrode_info, epoch_key,
        multitaper_parameter_name, group_name)
    save_pairwise_spectral_granger(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name)
    save_partial_directed_coherence(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name)
    save_canonical_coherence(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name)


def save_ERP(epoch_key, ERP, multitaper_parameter_name, group_name):
    logger.info('...saving ERP')
    group = '{0}/{1}/ERP'.format(
        multitaper_parameter_name, group_name)
    save_xarray(PROCESSED_DATA_DIR, epoch_key, ERP, group)


def save_power(
        c, tetrode_info, epoch_key,
        multitaper_parameter_name, group_name):
    logger.info('...saving power')
    dimension_names = ['time', 'frequency', 'tetrode']
    data_vars = {
        'power': (dimension_names, c.power())}
    coordinates = {
        'time': _center_time(c.time),
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode': tetrode_info.tetrode_id.values,
        'brain_area': ('tetrode', tetrode_info.area.tolist()),
    }
    group = '{0}/{1}/power'.format(
        multitaper_parameter_name, group_name)
    save_xarray(PROCESSED_DATA_DIR,
                epoch_key, xr.Dataset(data_vars, coords=coordinates), group)


def save_coherence(
        c, tetrode_info, epoch_key,
        multitaper_parameter_name, group_name):
    logger.info('...saving coherence')
    dimension_names = ['time', 'frequency', 'tetrode1', 'tetrode2']
    data_vars = {
        'coherence_magnitude': (dimension_names, c.coherence_magnitude())}
    coordinates = {
        'time': _center_time(c.time),
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode1': tetrode_info.tetrode_id.values,
        'tetrode2': tetrode_info.tetrode_id.values,
        'brain_area1': ('tetrode1', tetrode_info.area.tolist()),
        'brain_area2': ('tetrode2', tetrode_info.area.tolist()),
    }
    group = '{0}/{1}/coherence_magnitude'.format(
        multitaper_parameter_name, group_name)
    save_xarray(PROCESSED_DATA_DIR,
                epoch_key, xr.Dataset(data_vars, coords=coordinates), group)


def save_pairwise_spectral_granger(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name):
    logger.info('...saving pairwise spectral granger')
    dimension_names = ['time', 'frequency', 'tetrode1', 'tetrode2']
    data_vars = {'pairwise_spectral_granger_prediction': (
        dimension_names, c.pairwise_spectral_granger_prediction())}
    coordinates = {
        'time': _center_time(c.time),
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode1': tetrode_info.tetrode_id.values,
        'tetrode2': tetrode_info.tetrode_id.values,
        'brain_area1': ('tetrode1', tetrode_info.area.tolist()),
        'brain_area2': ('tetrode2', tetrode_info.area.tolist()),
    }
    group = '{0}/{1}/pairwise_spectral_granger_prediction'.format(
        multitaper_parameter_name, group_name)
    save_xarray(PROCESSED_DATA_DIR,
                epoch_key, xr.Dataset(data_vars, coords=coordinates), group)


def save_partial_directed_coherence(
        c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name):
    logger.info('...saving partial directed coherence')
    dimension_names = ['time', 'frequency', 'tetrode1', 'tetrode2']
    data_vars = {'partial_directed_coherence': (
        dimension_names, c.partial_directed_coherence())}
    coordinates = {
        'time': _center_time(c.time),
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'tetrode1': tetrode_info.tetrode_id.values,
        'tetrode2': tetrode_info.tetrode_id.values,
        'brain_area1': ('tetrode1', tetrode_info.area.tolist()),
        'brain_area2': ('tetrode2', tetrode_info.area.tolist()),
    }
    group = '{0}/{1}/partial_directed_coherence'.format(
        multitaper_parameter_name, group_name)
    save_xarray(PROCESSED_DATA_DIR,
                epoch_key, xr.Dataset(data_vars, coords=coordinates), group)


def save_canonical_coherence(
    c, tetrode_info, epoch_key, multitaper_parameter_name,
        group_name):
    logger.info('...saving canonical_coherence')
    canonical_coherence, area_labels = c.canonical_coherence(
        tetrode_info.area.tolist())
    dimension_names = ['time', 'frequency', 'brain_area1', 'brain_area2']
    data_vars = {
        'canonical_coherence': (dimension_names, canonical_coherence)}
    coordinates = {
        'time': _center_time(c.time),
        'frequency': c.frequencies + np.diff(c.frequencies)[0] / 2,
        'brain_area1': area_labels,
        'brain_area2': area_labels,
    }
    group = '{0}/{1}/canonical_coherence'.format(
        multitaper_parameter_name, group_name)
    save_xarray(PROCESSED_DATA_DIR,
                epoch_key, xr.Dataset(data_vars, coords=coordinates), group)


def save_group_delay(c, m, frequency_bands, tetrode_info, epoch_key,
                     multitaper_parameter_name, group_name):
    logger.info('...saving group delay')
    group_delay = np.array(
        [c.group_delay(frequency_bands[frequency_band],
                       frequency_resolution=m.frequency_resolution)
         for frequency_band in frequency_bands])

    dimension_names = ['frequency_band', 'time', 'tetrode1', 'tetrode2']
    data_vars = {
        'delay': (dimension_names, group_delay[:, 0, ...]),
        'slope': (dimension_names, group_delay[:, 1, ...]),
        'r_value': (dimension_names, group_delay[:, 2, ...])}

    coordinates = {
        'time': _center_time(c.time),
        'frequency_band': list(frequency_bands.keys()),
        'tetrode1': tetrode_info.tetrode_id.values,
        'tetrode2': tetrode_info.tetrode_id.values,
        'brain_area1': ('tetrode1', tetrode_info.area.tolist()),
        'brain_area2': ('tetrode2', tetrode_info.area.tolist()),
    }
    group = '{0}/{1}/group_delay'.format(
        multitaper_parameter_name, group_name)
    save_xarray(PROCESSED_DATA_DIR,
                epoch_key, xr.Dataset(data_vars, coords=coordinates), group)


def match_frequency_resolution(lfps, parameters, time_window_duration=2.0):
    desired_half_bandwidth = (parameters['time_halfbandwidth_product'] /
                              parameters['time_window_duration'])
    return time_window_duration * desired_half_bandwidth


def _center_time(time):
    time_diff = np.diff(time)[0] if np.diff(time).size > 0 else 0
    return time + time_diff / 2


def _get_ripple_times(df):
    '''Retrieves the ripple times from the ripple_info dataframe'''
    return (df.loc[:, ('ripple_start_time', 'ripple_end_time')]
            .values.tolist())


def detect_epoch_ripples(
        epoch_key, animals, sampling_frequency,
        position_info=None, brain_areas=_BRAIN_AREAS,
        minimum_duration=np.timedelta64(15, 'ms'),
        zscore_threshold=2,
        close_ripple_threshold=np.timedelta64(0, 's'),
        detector=Kay_ripple_detector):
    '''Returns a list of tuples containing the start and end times of
    ripples. Candidate ripples are computed via the ripple detection
    function and then filtered to exclude ripples where the animal was
    still moving.
    '''
    logger.info('Detecting ripples')

    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    brain_areas = [brain_areas] if isinstance(
        brain_areas, str) else brain_areas
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    if 'CA1' in brain_areas:
        is_brain_areas = is_brain_areas & tetrode_info.descrip.isin(['riptet'])
    logger.debug(tetrode_info[is_brain_areas]
                 .loc[:, ['area', 'depth', 'descrip']])
    tetrode_keys = tetrode_info[is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, animals)
    time = lfps.index
    if position_info is None:
        position_info = get_interpolated_position_dataframe(
            epoch_key, animals)
    speed = position_info.speed

    return detector(
        time, lfps.values, speed.values, sampling_frequency,
        minimum_duration=minimum_duration, zscore_threshold=zscore_threshold,
        close_ripple_threshold=close_ripple_threshold)


def decode_ripple_sorted_spikes(epoch_key, animals, ripple_times,
                                position_info, neuron_info,
                                sampling_frequency=1500,
                                n_position_bins=61):
    '''Labels the ripple by category

    Parameters
    ----------
    epoch_key : 3-element tuple
        Specifies which epoch to run.
        (Animal short name, day, epoch_number)
    animals : list of named-tuples
        Tuples give information to convert from the animal short name
        to a data directory
    ripple_times : list of 2-element tuples
        The first element of the tuple is the start time of the ripple.
        Second element of the tuple is the end time of the ripple
    sampling_frequency : int, optional
        Sampling frequency of the spikes
    n_position_bins : int, optional
        Number of bins for the linear distance

    Returns
    -------
    ripple_info : pandas dataframe
        Dataframe containing the categories for each ripple
        and the probability of that category

    '''
    logger.info('Decoding ripples')
    # Include only CA1 neurons with spikes
    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    neuron_info = pd.merge(tetrode_info, neuron_info.copy(),
                           on=['animal', 'day', 'epoch',
                               'tetrode_number', 'area'],
                           how='right', right_index=True).set_index(
        neuron_info.index)
    neuron_info = neuron_info[
        neuron_info.area.isin(['CA1', 'iCA1', 'CA3']) &
        (neuron_info.numspikes > 0)]
    logger.debug(neuron_info.loc[:, ['area', 'numspikes']])
    position_info['lagged_linear_distance'] = (
        position_info.linear_distance.shift(1))

    # Train on when the rat is moving
    spikes_data = [get_spike_indicator_dataframe(neuron_key, animals)
                   for neuron_key in neuron_info.index]

    ripple_indicator = get_ripple_indicator(epoch_key, animals, ripple_times)
    train_position_info = position_info.loc[
        ~ripple_indicator & position_info.is_correct]
    train_spikes_data = np.stack([spikes_datum.loc[train_position_info.index]
                                  for spikes_datum in spikes_data], axis=0)
    decoder = SortedSpikeDecoder(
        position=train_position_info.linear_distance.values,
        lagged_position=train_position_info.lagged_linear_distance.values,
        spikes=train_spikes_data,
        trajectory_direction=train_position_info.task.values
    ).fit()

    test_spikes = _get_ripple_spikes(
        spikes_data, ripple_times, sampling_frequency)
    results = [decoder.predict(ripple_spikes, time.total_seconds())
               for ripple_spikes, time in test_spikes]
    return summarize_replay_results(
        results, ripple_times, position_info, epoch_key)


def decode_ripple_clusterless(epoch_key, animals, ripple_times,
                              position_info=None,
                              sampling_frequency=1500,
                              n_position_bins=61,
                              place_std_deviation=None,
                              mark_std_deviation=20,
                              confidence_threshold=0.8,
                              mark_names=_MARKS,
                              brain_areas=_BRAIN_AREAS):
    logger.info('Decoding ripples')
    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)
    brain_areas = [brain_areas] if isinstance(
        brain_areas, str) else brain_areas
    is_brain_areas = tetrode_info.area.isin(brain_areas)
    brain_areas_tetrodes = tetrode_info[
        is_brain_areas &
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False) &
        ~tetrode_info.descrip.str.startswith('Ref').fillna(False)]
    logger.debug(brain_areas_tetrodes.loc[:, ['area', 'depth', 'descrip']])

    if mark_names is None:
        # Use all available mark dimensions
        mark_names = get_multiunit_indicator_dataframe(
            brain_areas_tetrodes.index[0], animals).columns.tolist()
        mark_names = [mark_name for mark_name in mark_names
                      if mark_name not in ['x_position', 'y_position']]

    marks = [(get_multiunit_indicator_dataframe(tetrode_key, animals)
              .loc[:, mark_names])
             for tetrode_key in brain_areas_tetrodes.index]
    marks = [tetrode_marks for tetrode_marks in marks
             if (tetrode_marks.loc[position_info.speed > 4, :].dropna()
                 .shape[0]) != 0]

    if position_info is None:
        position_info = get_interpolated_position_dataframe(epoch_key, animals)
    else:
        position_info = position_info.copy()

    position_info['lagged_linear_distance'] = (
        position_info.linear_distance.shift(1))
    KEEP_COLUMNS = ['linear_distance', 'lagged_linear_distance', 'task',
                    'is_correct', 'turn', 'speed', 'head_direction']

    ripple_indicator = get_ripple_indicator(epoch_key, animals, ripple_times)
    train_position_info = position_info[KEEP_COLUMNS].loc[
        ~ripple_indicator & position_info.is_correct].dropna()

    training_marks = np.stack([
        tetrode_marks.loc[train_position_info.index, mark_names]
        for tetrode_marks in marks], axis=0)

    decoder = ClusterlessDecoder(
        position=train_position_info.linear_distance.values,
        lagged_position=train_position_info.lagged_linear_distance.values,
        trajectory_direction=train_position_info.task.values,
        spike_marks=training_marks,
        n_position_bins=n_position_bins,
        place_std_deviation=place_std_deviation,
        mark_std_deviation=mark_std_deviation,
        replay_speedup_factor=16,
        confidence_threshold=confidence_threshold,
    ).fit()

    test_marks = _get_ripple_marks(marks, ripple_times, sampling_frequency)
    logger.info('Predicting replay types')
    results = [decoder.predict(ripple_marks, time.total_seconds())
               for ripple_marks, time in test_marks]

    return summarize_replay_results(
        results, ripple_times, position_info, epoch_key)


def _get_ripple_marks(marks, ripple_times, sampling_frequency):
    mark_ripples = [reshape_to_segments(
        tetrode_marks, ripple_times, sampling_frequency=sampling_frequency,
        axis=0)
        for tetrode_marks in marks]

    return [(np.stack([df.loc[ripple_number, :].values
                       for df in mark_ripples], axis=0),
             mark_ripples[0].loc[ripple_number, :]
             .index.get_level_values('time'))
            for ripple_number in ripple_times.index]


def _get_ripple_spikes(spikes_data, ripple_times, sampling_frequency):
    '''Given the ripple times, extract the spikes within the ripple
    '''
    spike_ripples = [reshape_to_segments(
        spikes_datum, ripple_times, axis=0,
        sampling_frequency=sampling_frequency)
        for spikes_datum in spikes_data]

    return [
        (np.stack([df.loc[ripple_number, :].values
                   for df in spike_ripples], axis=0).squeeze(),
         spike_ripples[0].loc[ripple_number, :].index.get_level_values('time'))
        for ripple_number in ripple_times.index]


def summarize_replay_results(results, ripple_times, position_info,
                             epoch_key):
    '''Summary statistics for decoded replays.

    Parameters
    ----------
    posterior_density : list of arrays
    test_spikes : array_like
    ripple_times : list of tuples
    state_names : list of str
    position_info : pandas DataFrame

    Returns
    -------
    replay_info : pandas dataframe
    decision_state_probability : array_like
    posterior_density : xarray DataArray

    '''
    replay_info = ripple_times.copy()

    # Includes information about the animal, day, epoch in index
    (replay_info['animal'], replay_info['day'],
     replay_info['epoch']) = epoch_key
    replay_info = replay_info.reset_index()

    replay_info['ripple_duration'] = (
        replay_info['end_time'] - replay_info['start_time'])

    # Add decoded states and probability of state
    replay_info['predicted_state'] = [
        result.predicted_state() for result in results]
    replay_info['predicted_state_probability'] = [
        result.predicted_state_probability() for result in results]

    replay_info = pd.concat(
        (replay_info,
         replay_info.predicted_state.str.split('-', expand=True)
         .rename(columns={0: 'replay_task',
                          1: 'replay_order'})
         ), axis=1)

    # When in the session does the ripple occur (early, middle, late)
    replay_info['session_time'] = pd.Categorical(
        _ripple_session_time(ripple_times, position_info.index), ordered=True,
        categories=['early', 'middle', 'late'])

    # Add stats about spikes
    replay_info['number_of_unique_spiking'] = [
        _num_unique_spiking(result.spikes) for result in results]
    replay_info['number_of_spikes'] = [_num_total_spikes(result.spikes)
                                       for result in results]

    # Include animal position information
    replay_info = pd.concat(
        (replay_info.set_index('ripple_number'),
         position_info.loc[replay_info.start_time]
         .set_index(ripple_times.index)), axis=1).reset_index()

    # Determine whether ripple is heading towards or away from animal's
    # position
    posterior_density = xr.concat(
        [result.results.posterior_density for result in results],
        dim=replay_info.index)
    replay_info['replay_motion'] = _get_replay_motion(
        replay_info, posterior_density)

    decision_state_probability = xr.concat(
        [result.state_probability().unstack().to_xarray().rename(
            'decision_state_probability')
         for result in results], dim=replay_info.index)

    return (replay_info, decision_state_probability,
            posterior_density)


def _num_unique_spiking(spikes):
    '''Number of units that spike per ripple
    '''
    if spikes.ndim > 2:
        return np.sum(~np.isnan(spikes), axis=(1, 2)).nonzero()[0].size
    else:
        return spikes.sum(axis=0).nonzero()[0].size


def _num_total_spikes(spikes):
    '''Total number of spikes per ripple
    '''
    if spikes.ndim > 2:
        return np.any(~np.isnan(spikes), axis=2).sum()
    else:
        return int(spikes.sum())


def _ripple_session_time(ripple_times, session_time):
    '''Categorize the ripples by the time in the session in which they
    occur.

    This function trichotimizes the session time into early session,
    middle session, and late session and classifies the ripple by the most
    prevelant category.
    '''
    session_time_categories = pd.Series(
        pd.cut(
            session_time, 3,
            labels=['early', 'middle', 'late'], precision=4),
        index=session_time)
    return pd.Series(
        [(session_time_categories.loc[ripple_start:ripple_end]
          .value_counts().argmax())
         for ripple_start, ripple_end
         in ripple_times.itertuples(index=False)],
        index=ripple_times.index, name='session_time',
        dtype=session_time_categories.dtype)


def _get_replay_motion_from_rows(ripple_times, posterior_density,
                                 distance_measure='linear_distance'):
    '''

    Parameters
    ----------
    ripple_info : pandas dataframe row
    posterior_density : array, shape (n_time, n_position_bins)
    state_names : list of str, shape (n_states,)
    place_bin_centers : array (n_position_bins)

    Returns
    -------
    is_away : array of str

    '''
    posterior_density = posterior_density.sum('state').dropna('time')
    replay_position = posterior_density.position.values[
        posterior_density.argmax('position').values]
    animal_position = ripple_times[distance_measure]
    replay_distance_from_animal_position = np.abs(
        replay_position - animal_position)
    is_away = linregress(
        posterior_density.get_index('time').values,
        replay_distance_from_animal_position).slope > 0
    return np.where(is_away, 'Away', 'Towards')


def _get_replay_motion(ripple_times, posterior_density,
                       distance_measure='linear_distance'):
    '''Motion of the replay relative to the current position of the animal.
    '''
    return np.array(
        [_get_replay_motion_from_rows(row, density, distance_measure)
         for (_, row), density
         in zip(ripple_times.iterrows(), posterior_density)]).squeeze()


def _subtract_event_related_potential(df):
    return df.apply(lambda x: x - df.mean(axis=1), raw=True)


def is_overlap(band1, band2):
    return (band1[0] < band2[1]) & (band2[0] < band1[1])


def get_ripple_indicator(epoch_key, animals, ripple_times):
    time = get_trial_time(epoch_key, animals)
    ripple_indicator = pd.Series(np.zeros_like(time, dtype=bool), index=time)
    for _, start_time, end_time in ripple_times.itertuples():
        ripple_indicator[start_time:end_time] = True

    return ripple_indicator


def theta_filter(sampling_frequency):
    ORDER = 1500
    nyquist = 0.5 * sampling_frequency
    TRANSITION_BAND = 2
    THETA_BAND = [6, 10]
    desired = [0, THETA_BAND[0] - TRANSITION_BAND, THETA_BAND[0],
               THETA_BAND[1], THETA_BAND[1] + TRANSITION_BAND, nyquist]
    return remez(ORDER, desired, [0, 1, 0], Hz=sampling_frequency), 1.0


def get_hippocampal_theta(epoch_key, animals, sampling_frequency):
    tetrode_info = make_tetrode_dataframe(animals).xs(
        epoch_key, drop_level=False)

    tetrode_key = tetrode_info.loc[tetrode_info.descrip == 'CA1Ref'].index
    lfp = get_LFPs(tetrode_key, animals)
    filter_numerator, filter_denominator = theta_filter(sampling_frequency)
    theta_filtered = filtfilt(
        filter_numerator, filter_denominator, lfp.values, axis=0)
    analytic_signal = hilbert(theta_filtered, axis=0)
    instantaneous_phase = np.angle(analytic_signal)
    instantaneous_amplitude = np.abs(analytic_signal)

    return pd.DataFrame(
        np.concatenate((instantaneous_phase, instantaneous_amplitude), axis=1),
        index=lfp.index,
        columns=['instantaneous_phase', 'instantaneous_amplitude'])


def adjusted_coherence_magnitude(spikes, sampling_frequency, m, c):
    '''

    Parameters
    ----------
    spikes : ndarray, shape (n_time, n_trials, n_signals)
    sampling_frequency : float
    **multitaper_params : kwargs

    Returns
    -------
    adjusted_coherence_magnitude : ndarray, shape (n_time, n_frequencies,
                                                   n_signals, n_signals)

    '''

    dt = 1.0 / sampling_frequency

    # Multiply by sampling_frequency?
    average_rate = _sliding_window(
        spikes, m.n_time_samples_per_window,
        m.n_time_samples_per_step, axis=0).mean(axis=(1, 3))

    power = c.power()
    n_time_windows = average_rate.shape[0]

    adjustment_factor = np.stack([
        coherence_rate_adjustment(average_rate[0], average_rate[time_ind],
                                  power[0, ...], dt=dt)
        for time_ind in range(n_time_windows)
    ], axis=0)[..., np.newaxis]

    return np.abs(c.coherency()
                  * adjustment_factor *
                  adjustment_factor.swapaxes(-1, -2))


def get_ripple_locked_spikes(neuron_keys, ripple_times, animals,
                             sampling_frequency=1, window_offset=(-0.5, 0.5)):
    spikes = get_all_spike_indicators(neuron_keys, animals)
    return reshape_to_segments(
        spikes, ripple_times, window_offset, sampling_frequency).fillna(0)
