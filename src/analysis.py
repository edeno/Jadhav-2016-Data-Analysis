'''Higher level functions for analyzing the data

'''
from copy import deepcopy
from functools import partial
from logging import getLogger
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import linregress

import xarray as xr

from .data_processing import (get_interpolated_position_dataframe,
                              get_LFP_dataframe, get_mark_indicator_dataframe,
                              get_spike_indicator_dataframe,
                              make_neuron_dataframe, make_tetrode_dataframe,
                              reshape_to_segments, save_xarray)
from .ripple_decoding import ClusterlessDecoder, SortedSpikeDecoder
from .ripple_detection.detectors import Kay_ripple_detector
from .spectral.connectivity import Connectivity
from .spectral.transforms import Multitaper, _sliding_window
from .spectral.statistics import (fisher_z_transform,
                                  get_normal_distribution_p_values,
                                  coherence_bias, coherence_rate_adjustment)
from .spike_train import perievent_time_spline_estimate, cross_correlate

logger = getLogger(__name__)


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


def ripple_locked_firing_rate_change(ripple_times, neuron_info, animals,
                                     sampling_frequency,
                                     window_offset=None,
                                     n_boot_samples=None,
                                     formula='bs(time, df=5)'):

    estimate = []
    for neuron_key in neuron_info.index:
        spikes = get_spike_indicator_dataframe(neuron_key, animals)
        ripple_locked_spikes = reshape_to_segments(
            spikes, ripple_times, sampling_frequency=sampling_frequency,
            window_offset=window_offset)
        time = ripple_locked_spikes.index.get_level_values('time').values
        trial_id = (ripple_locked_spikes.index
                    .get_level_values('segment_number').values)
        estimate.append(
            perievent_time_spline_estimate(
                ripple_locked_spikes.values.squeeze(),
                time, sampling_frequency, formula=formula,
                n_boot_samples=n_boot_samples, trial_id=trial_id))

    return xr.concat(estimate, dim=neuron_info.neuron_id)


def ripple_cross_correlation(ripple_times, neuron_info, animals,
                             sampling_frequency, window_offset=None):

    before_ripple, after_ripple = [], []

    for neuron_key in neuron_info.index:
        spikes = get_spike_indicator_dataframe(neuron_key, animals)
        ripple_locked_spikes = reshape_to_segments(
            spikes, ripple_times, sampling_frequency=sampling_frequency,
            window_offset=window_offset).unstack(level=0)
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
                spikes, ripple_times, sampling_frequency=sampling_frequency,
                window_offset=window_offset).unstack(level=0))
    m = Multitaper(np.stack(ripple_locked_spikes, axis=-1),
                   **multitaper_parameters,
                   start_time=ripple_locked_spikes[0].index.values[0])
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


def compare_spike_coherence(firing_rate1, firing_rate2, power1, coherency1,
                            coherency2, sampling_frequency, n_trials1,
                            n_trials2, n_tapers=1):
    dt = 1.0 / sampling_frequency
    adjustment = coherence_rate_adjustment(
        firing_rate1, firing_rate2, power1, dt=dt)

    adjusted_coherency1 = (adjustment.rename({'neuron1': 'neuron2'}) *
                           adjustment * coherency1).transpose(
                           'frequency', 'neuron1', 'neuron2')

    coherence_difference = np.abs(coherency2) - np.abs(adjusted_coherency1)
    bias1 = coherence_bias(n_trials1 * n_tapers)
    bias2 = coherence_bias(n_trials2 * n_tapers)
    coherence_z_difference = fisher_z_transform(
        coherency2.values, bias2, adjusted_coherency1.values, bias1)
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
        'n_trials1': n_trials1,
        'n_trials2': n_trials2,
        'n_tapers': n_tapers,
    }

    return xr.Dataset(data_vars, coords, attrs).drop('time')


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
        FREQUENCY_BANDS, multitaper_parameter_name='',
        group_name='all_ripples'):
    n_lfps = len(lfps)
    n_pairs = int(n_lfps * (n_lfps - 1) / 2)
    params = deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')
    reshape_to_trials = partial(
        reshape_to_segments,
        sampling_frequency=params['sampling_frequency'],
        window_offset=window_of_interest, concat_axis=1)

    logger.info('Computing ripple-triggered {multitaper_parameter_name} '
                'for {num_pairs} pairs of electrodes'.format(
                    multitaper_parameter_name=multitaper_parameter_name,
                    num_pairs=n_pairs))

    ripple_ERP = pd.concat({
        tetrode_info.loc[lfp_name].tetrode_id: reshape_to_trials(
            lfps[lfp_name], ripple_times).mean(axis=1)
        for lfp_name in lfps}, axis=1)

    ripple_locked_lfps = pd.Panel({
        lfp_name: _subtract_event_related_potential(
            reshape_to_trials(lfps[lfp_name], ripple_times))
        for lfp_name in lfps}).dropna(axis=2)

    m = Multitaper(
        np.rollaxis(ripple_locked_lfps.values, 0, 3),
        **params,
        start_time=ripple_locked_lfps.major_axis.min())
    c = Connectivity.from_multitaper(m)

    save_ERP(epoch_key, ripple_ERP, multitaper_parameter_name, group_name)

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
    save_xarray(epoch_key,
                xr.DataArray(ERP).rename({'dim_1': 'tetrode'}),
                group)


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
    save_xarray(
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
    save_xarray(
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
    save_xarray(
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
    save_xarray(
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
    save_xarray(
        epoch_key, xr.Dataset(data_vars, coords=coordinates), group)


def save_group_delay(c, m, FREQUENCY_BANDS, tetrode_info, epoch_key,
                     multitaper_parameter_name, group_name):
    logger.info('...saving group delay')
    group_delay = np.array(
        [c.group_delay(FREQUENCY_BANDS[frequency_band],
                       frequency_resolution=m.frequency_resolution)
         for frequency_band in FREQUENCY_BANDS])

    dimension_names = ['frequency_band', 'time', 'tetrode1', 'tetrode2']
    data_vars = {
        'delay': (dimension_names, group_delay[:, 0, ...]),
        'slope': (dimension_names, group_delay[:, 1, ...]),
        'r_value': (dimension_names, group_delay[:, 2, ...])}

    coordinates = {
        'time': _center_time(c.time),
        'frequency_band': list(FREQUENCY_BANDS.keys()),
        'tetrode1': tetrode_info.tetrode_id.values,
        'tetrode2': tetrode_info.tetrode_id.values,
        'brain_area1': ('tetrode1', tetrode_info.area.tolist()),
        'brain_area2': ('tetrode2', tetrode_info.area.tolist()),
    }
    group = '{0}/{1}/group_delay'.format(
        multitaper_parameter_name, group_name)
    save_xarray(
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


def detect_epoch_ripples(epoch_key, animals, sampling_frequency):
    '''Returns a list of tuples containing the start and end times of
    ripples. Candidate ripples are computed via the ripple detection
    function and then filtered to exclude ripples where the animal was
    still moving.
    '''
    logger.info('Detecting ripples')
    tetrode_info = (
        make_tetrode_dataframe(animals)
        .loc[epoch_key]
        .set_index(['animal', 'day', 'epoch', 'tetrode_number'],
                   drop=False))
    # Get cell-layer CA1, iCA1 LFPs
    is_hippocampal = (tetrode_info.area.isin(['CA1', 'iCA1', 'CA3']) &
                      (tetrode_info.descrip.isin(['riptet']) |
                       tetrode_info.validripple))
    logger.debug(tetrode_info[is_hippocampal]
                 .loc[:, ['area', 'depth', 'descrip']])
    tetrode_keys = tetrode_info[is_hippocampal].index.tolist()
    hippocampus_lfps = pd.concat(
        [get_LFP_dataframe(tetrode_key, animals)
         for tetrode_key in tetrode_keys], axis=1)
    speed = get_interpolated_position_dataframe(
        epoch_key, animals).speed.values
    time = hippocampus_lfps.index.values
    return Kay_ripple_detector(
        time, hippocampus_lfps.values, speed, sampling_frequency)


def decode_ripple_sorted_spikes(epoch_key, animals, ripple_times,
                                sampling_frequency=1500,
                                n_place_bins=61):
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
    n_place_bins : int, optional
        Number of bins for the linear distance

    Returns
    -------
    ripple_info : pandas dataframe
        Dataframe containing the categories for each ripple
        and the probability of that category

    '''
    logger.info('Decoding ripples')
    # Include only CA1 neurons with spikes
    neuron_info = make_neuron_dataframe(animals)[
        epoch_key].dropna()
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_key]
    neuron_info = pd.merge(tetrode_info, neuron_info,
                           on=['animal', 'day', 'epoch',
                               'tetrode_number', 'area'],
                           how='right', right_index=True).set_index(
        neuron_info.index)
    neuron_info = neuron_info[
        neuron_info.area.isin(['CA1', 'iCA1', 'CA3']) &
        (neuron_info.numspikes > 0) &
        ~neuron_info.descrip.str.endswith('Ref').fillna(False)]
    logger.debug(neuron_info.loc[:, ['area', 'numspikes']])

    # Train on when the rat is moving
    position_info = get_interpolated_position_dataframe(epoch_key, animals)
    spikes_data = [get_spike_indicator_dataframe(neuron_key, animals)
                   for neuron_key in neuron_info.index]

    # Make sure there are spikes in the training data times. Otherwise
    # exclude that neuron
    spikes_data = [spikes_datum for spikes_datum in spikes_data
                   if spikes_datum[
                       position_info.speed > 4].sum().values > 0]

    train_position_info = position_info.query('speed > 4')
    train_spikes_data = [spikes_datum[position_info.speed > 4]
                         for spikes_datum in spikes_data]
    decoder = SortedSpikeDecoder(
        position=train_position_info.linear_distance.values,
        spikes=np.stack(train_spikes_data, axis=0),
        trajectory_direction=train_position_info.trajectory_direction.values
    )

    test_spikes = _get_ripple_spikes(
        spikes_data, ripple_times.values, sampling_frequency)
    results = [decoder.predict(ripple_spikes)
               for ripple_spikes in test_spikes]
    return summarize_replay_results(
        results, ripple_times, position_info, epoch_key)


def decode_ripple_clusterless(epoch_key, animals, ripple_times,
                              sampling_frequency=1500,
                              n_place_bins=61,
                              place_std_deviation=None,
                              mark_std_deviation=20):
    logger.info('Decoding ripples')
    mark_variables = ['channel_1_max', 'channel_2_max', 'channel_3_max',
                      'channel_4_max']
    tetrode_info = (
        make_tetrode_dataframe(animals)
        .loc[epoch_key]
        .set_index(['animal', 'day', 'epoch', 'tetrode_number'],
                   drop=False))
    # Get cell-layer CA1, iCA1 LFPs
    is_hippocampal = tetrode_info.area.isin(['CA1', 'iCA1', 'CA3'])
    hippocampal_tetrodes = tetrode_info[
        is_hippocampal &
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False) &
        ~tetrode_info.descrip.str.startswith('Ref').fillna(False)]
    logger.debug(hippocampal_tetrodes.loc[:, ['area', 'depth', 'descrip']])

    position_info = get_interpolated_position_dataframe(epoch_key, animals)

    marks = [(get_mark_indicator_dataframe(tetrode_key, animals)
              .loc[:, mark_variables])
             for tetrode_key in hippocampal_tetrodes.index]
    marks = [tetrode_marks for tetrode_marks in marks
             if (tetrode_marks.loc[position_info.speed > 4, :].dropna()
                 .shape[0]) != 0]

    train_position_info = position_info.query('speed > 4')

    training_marks = np.stack([
        tetrode_marks.loc[train_position_info.index, mark_variables]
        for tetrode_marks in marks], axis=0)

    decoder = ClusterlessDecoder(
        train_position_info.linear_distance.values,
        train_position_info.trajectory_direction.values,
        training_marks
    ).fit()

    test_marks = _get_ripple_marks(
        marks, ripple_times.values, sampling_frequency)
    logger.info('Predicting replay types')
    results = [decoder.predict(ripple_marks, time)
               for ripple_marks, time in test_marks]

    return summarize_replay_results(
        results, ripple_times, position_info, epoch_key)


def _get_ripple_marks(marks, ripple_times, sampling_frequency):
    mark_ripples = [reshape_to_segments(
        tetrode_marks, ripple_times,
        concat_axis=0, sampling_frequency=sampling_frequency)
        for tetrode_marks in marks]

    return [(np.stack([df.loc[ripple_ind + 1, :].values
                       for df in mark_ripples], axis=0),
             mark_ripples[0].loc[ripple_ind + 1, :]
             .index.get_level_values('time'))
            for ripple_ind in np.arange(len(ripple_times))]


def _get_ripple_spikes(spikes_data, ripple_times, sampling_frequency):
    '''Given the ripple times, extract the spikes within the ripple
    '''
    spike_ripples_df = [reshape_to_segments(
        spikes_datum, ripple_times,
        concat_axis=1, sampling_frequency=sampling_frequency)
        for spikes_datum in spikes_data]

    return [np.vstack([df.iloc[:, ripple_ind].dropna().values
                       for df in spike_ripples_df]).T
            for ripple_ind in np.arange(len(ripple_times))]


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
    replay_info.reset_index().set_index(
        ['animal', 'day', 'epoch', 'ripple_number'], inplace=True)

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
    replay_info['session_time'] = _ripple_session_time(
        replay_info, position_info.index)

    # Add stats about spikes
    replay_info['number_of_unique_spiking'] = [
        _num_unique_spiking(result.spikes) for result in results]
    replay_info['number_of_spikes'] = [_num_total_spikes(result.spikes)
                                       for result in results]

    # Include animal position information
    replay_info = pd.concat(
        [replay_info,
         position_info.loc[replay_info.start_time]
         .drop('trajectory_category_ind', axis=1)
         .set_index(replay_info.index)
         ], axis=1)

    # Determine whether ripple is heading towards or away from animal's
    # position
    posterior_density = xr.concat(
        [result.posterior_density for result in results],
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
         in ripple_times.loc[:, ['start_time', 'end_time']].values],
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
    max_state_ind = int(posterior_density
                        .dropna('time').sum('position')
                        .isel(time=-1).argmax())
    posterior_density = posterior_density.isel(
        state=max_state_ind).dropna('time')
    replay_position = posterior_density.position.values[
        posterior_density.argmax('position')]
    animal_position = ripple_times[distance_measure]
    replay_distance_from_animal_position = np.abs(
        replay_position - animal_position)
    is_away = linregress(
        posterior_density.time.values,
        replay_distance_from_animal_position).slope > 0
    return np.where(is_away, 'away', 'towards')


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
