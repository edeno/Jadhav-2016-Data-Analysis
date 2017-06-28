'''Higher level functions for analyzing the data

'''
from copy import deepcopy
from functools import partial, wraps
from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from dask import local, compute, delayed

from .data_processing import (get_interpolated_position_dataframe,
                              get_LFP_dataframe,
                              get_mark_indicator_dataframe,
                              get_spike_indicator_dataframe,
                              make_neuron_dataframe,
                              make_tetrode_dataframe, reshape_to_segments,
                              save_xarray)
from .ripple_decoding import (combined_likelihood,
                              estimate_marked_encoding_model,
                              estimate_sorted_spike_encoding_model,
                              estimate_state_transition, get_bin_centers,
                              predict_state, set_initial_conditions)
from .ripple_detection import Kay_method
from .spectral.connectivity import Connectivity
from .spectral.transforms import Multitaper

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

    ripple_locked_lfps = pd.Panel({
        lfp_name: _subtract_event_related_potential(
            reshape_to_trials(lfps[lfp_name], ripple_times))
        for lfp_name in lfps})
    m = Multitaper(
        np.rollaxis(ripple_locked_lfps.values, 0, 3),
        **params,
        start_time=ripple_locked_lfps.major_axis.min())
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
    n_bands = len(FREQUENCY_BANDS)
    delay, slope, r_value = (
        np.zeros((c.time.size, n_bands, m.n_signals, m.n_signals)),) * 3

    for band_ind, frequency_band in enumerate(FREQUENCY_BANDS):
        (delay[:, band_ind, ...],
         slope[:, band_ind, ...],
         r_value[:, band_ind, ...]) = c.group_delay(
            FREQUENCY_BANDS[frequency_band],
            frequency_resolution=m.frequency_resolution)

    dimension_names = ['time', 'frequency_band', 'tetrode1', 'tetrode2']
    data_vars = {
        'delay': (dimension_names, delay),
        'slope': (dimension_names, slope),
        'r_value': (dimension_names, r_value)}
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


def detect_epoch_ripples(epoch_key, animals, sampling_frequency,
                         ripple_detection_function=Kay_method,
                         ripple_detection_kwargs={}, speed_threshold=4):
    '''Returns a list of tuples containing the start and end times of
    ripples. Candidate ripples are computed via the ripple detection
    function and then filtered to exclude ripples where the animal was
    still moving.
    '''
    logger.info('Detecting ripples')
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_key]
    # Get cell-layer CA1, iCA1 LFPs
    is_hippocampal = (tetrode_info.area.isin(['CA1', 'iCA1']) &
                      tetrode_info.descrip.isin(['riptet']))
    logger.debug(tetrode_info[is_hippocampal]
                 .loc[:, ['area', 'depth', 'descrip']])
    tetrode_keys = tetrode_info[is_hippocampal].index.tolist()
    CA1_lfps = [get_LFP_dataframe(tetrode_key, animals)
                for tetrode_key in tetrode_keys]
    candidate_ripple_times = ripple_detection_function(
        CA1_lfps, **ripple_detection_kwargs)
    return exclude_movement_during_ripples(
        candidate_ripple_times, epoch_key, animals, speed_threshold)


def decode_ripple_sorted_spikes(epoch_key, animals, ripple_times,
                                sampling_frequency=1500,
                                n_place_bins=49):
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
        neuron_info.area.isin(['CA1', 'iCA1']) &
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
    place_bin_edges = np.linspace(
        np.floor(position_info.linear_distance.min()),
        np.ceil(position_info.linear_distance.max()),
        n_place_bins + 1)
    place_bin_centers = get_bin_centers(
        place_bin_edges)

    logger.info('...Fitting encoding model')
    combined_likelihood_kwargs = estimate_sorted_spike_encoding_model(
        train_position_info, train_spikes_data, place_bin_centers)

    logger.info('...Fitting state transition model')
    state_transition = estimate_state_transition(
        train_position_info, place_bin_edges)

    logger.info('...Setting initial conditions')
    state_names = ['outbound_forward', 'outbound_reverse',
                   'inbound_forward', 'inbound_reverse']
    n_states = len(state_names)
    initial_conditions = set_initial_conditions(
        place_bin_edges, place_bin_centers, n_states)

    logger.info('...Decoding ripples')
    decoder_params = dict(
        initial_conditions=initial_conditions,
        state_transition=state_transition,
        likelihood_function=combined_likelihood,
        likelihood_kwargs=combined_likelihood_kwargs
    )
    test_spikes = _get_ripple_spikes(
        spikes_data, ripple_times, sampling_frequency)
    posterior_density = [predict_state(ripple_spikes, **decoder_params)
                         for ripple_spikes in test_spikes]
    return get_ripple_info(
        posterior_density, test_spikes, ripple_times,
        state_names, position_info.index, epoch_key)


def decode_ripple_clusterless(epoch_key, animals, ripple_times,
                              sampling_frequency=1500,
                              n_place_bins=61,
                              place_std_deviation=None,
                              mark_std_deviation=20,
                              scheduler=local.get_sync,
                              scheduler_kwargs={}):
    logger.info('Decoding ripples')
    tetrode_info = make_tetrode_dataframe(animals)[epoch_key]
    mark_variables = ['channel_1_max', 'channel_2_max', 'channel_3_max',
                      'channel_4_max']
    hippocampal_tetrodes = tetrode_info.loc[
        tetrode_info.area.isin(['CA1', 'iCA1']) &
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False), :]
    logger.debug(hippocampal_tetrodes.loc[:, ['area', 'depth', 'descrip']])

    position_variables = ['linear_distance', 'trajectory_direction',
                          'speed']
    position_info = (get_interpolated_position_dataframe(
        epoch_key, animals).loc[:, position_variables])

    marks = [(get_mark_indicator_dataframe(tetrode_key, animals)
              .loc[:, mark_variables])
             for tetrode_key in hippocampal_tetrodes.index]
    marks = [tetrode_marks for tetrode_marks in marks
             if (tetrode_marks.loc[position_info.speed > 4, :].dropna()
                 .shape[0]) != 0]

    train_position_info = position_info.query('speed > 4')

    place = _get_place(train_position_info)
    place_at_spike = [_get_place_at_spike(tetrode_marks,
                                          train_position_info)
                      for tetrode_marks in marks]
    training_marks = [_get_training_marks(tetrode_marks,
                                          train_position_info,
                                          mark_variables)
                      for tetrode_marks in marks]

    place_bin_edges = np.linspace(
        np.floor(position_info.linear_distance.min()),
        np.ceil(position_info.linear_distance.max()),
        n_place_bins + 1)
    place_bin_centers = get_bin_centers(place_bin_edges)

    if place_std_deviation is None:
        place_std_deviation = place_bin_edges[1] - place_bin_edges[0]

    logger.info('...Fitting encoding model')
    combined_likelihood_kwargs = estimate_marked_encoding_model(
        place_bin_centers, place, place_at_spike, training_marks,
        place_std_deviation=place_std_deviation,
        mark_std_deviation=mark_std_deviation)

    logger.info('...Fitting state transition model')
    state_transition = estimate_state_transition(
        train_position_info, place_bin_edges)

    logger.info('...Setting initial conditions')
    state_names = ['outbound_forward', 'outbound_reverse',
                   'inbound_forward', 'inbound_reverse']
    n_states = len(state_names)
    initial_conditions = set_initial_conditions(
        place_bin_edges, place_bin_centers, n_states)

    logger.info('...Decoding ripples')
    decoder_kwargs = dict(
        initial_conditions=initial_conditions,
        state_transition=state_transition,
        likelihood_function=combined_likelihood,
        likelihood_kwargs=combined_likelihood_kwargs
    )

    test_marks = _get_ripple_marks(
        marks, ripple_times, sampling_frequency)

    posterior_density = [
        delayed(predict_state, pure=True)(ripple_marks, **decoder_kwargs)
        for ripple_marks in test_marks]
    posterior_density = compute(
        *posterior_density, get=scheduler, **scheduler_kwargs)
    test_spikes = [np.mean(~np.isnan(marks), axis=2)
                   for marks in test_marks]

    return get_ripple_info(
        posterior_density, test_spikes, ripple_times,
        state_names, position_info.index, epoch_key)


def _convert_to_states(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        d = function(*args, **kwargs)
        return [d['Outbound'], d['Outbound'],
                d['Inbound'], d['Inbound']]
    return wrapper


@_convert_to_states
def _get_place(train_position_info, place_measure='linear_distance'):
    return {trajectory_direction: grouped.loc[:, place_measure].values
            for trajectory_direction, grouped
            in (train_position_info
                .groupby('trajectory_direction'))}


@_convert_to_states
def _get_place_at_spike(tetrode_marks, train_position_info,
                        place_measure='linear_distance'):
    return {trajectory_direction: (grouped.dropna()
                                   .loc[:, place_measure].values)
            for trajectory_direction, grouped
            in (tetrode_marks
                .join(train_position_info)
                .groupby('trajectory_direction'))}


@_convert_to_states
def _get_training_marks(tetrode_marks, train_position_info,
                        mark_variables):
    return {trajectory_direction: (grouped.dropna()
                                   .loc[:, mark_variables].values)
            for trajectory_direction, grouped
            in (tetrode_marks
                .join(train_position_info)
                .groupby('trajectory_direction'))}


def _get_ripple_marks(marks, ripple_times, sampling_frequency):
    mark_ripples = [reshape_to_segments(
        tetrode_marks, ripple_times,
        concat_axis=0, sampling_frequency=sampling_frequency)
        for tetrode_marks in marks]

    return [np.stack([df.loc[ripple_ind + 1, :].values
                      for df in mark_ripples], axis=1)
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


def exclude_movement_during_ripples(ripple_times, epoch_key, animals,
                                    speed_threshold):
    '''Excludes ripples where the head direction speed is greater than the
    speed threshold. Only looks at the start of the ripple to determine
    head movement speed for the ripple.
    '''
    position_df = get_interpolated_position_dataframe(
        epoch_key, animals)
    return [(ripple_start, ripple_end)
            for ripple_start, ripple_end in ripple_times
            if position_df.loc[
                ripple_start:ripple_end].speed.iloc[0] < speed_threshold]


def get_ripple_info(posterior_density, test_spikes, ripple_times,
                    state_names, session_time, epoch_key):
    '''Summary statistics for ripple categories

    Parameters
    ----------
    posterior_density : array_like
    test_spikes : array_like
    ripple_times : list of tuples
    state_names : list of str
    session_time : array_like

    Returns
    -------
    ripple_info : pandas dataframe
    decision_state_probability : array_like
    posterior_density : array_like
    state_names : list of str

    '''
    n_states = len(state_names)
    n_ripples = len(ripple_times)
    decision_state_probability = [
        _compute_decision_state_probability(density, n_states)
        for density in posterior_density]
    index = pd.MultiIndex.from_tuples(
        [(*epoch_key, ripple+1) for ripple in range(n_ripples)],
        names=['animal', 'day', 'epoch', 'ripple_number'])
    ripple_info = pd.DataFrame(
        [_compute_max_state(probability, state_names)
         for probability in decision_state_probability],
        columns=['ripple_trajectory', 'ripple_direction',
                 'ripple_state_probability'],
        index=index)
    ripple_info['ripple_start_time'] = np.asarray(ripple_times)[:, 0]
    ripple_info['ripple_end_time'] = np.asarray(ripple_times)[:, 1]
    ripple_info['number_of_unique_neurons_spiking'] = [
        _num_unique_neurons_spiking(spikes) for spikes in test_spikes]
    ripple_info['number_of_spikes'] = [_num_total_spikes(spikes)
                                       for spikes in test_spikes]
    ripple_info['session_time'] = _ripple_session_time(
        ripple_times, session_time)
    ripple_info['is_spike'] = ((ripple_info.number_of_spikes > 0)
                               .map({True: 'isSpike', False: 'noSpike'}))

    return (ripple_info, decision_state_probability,
            posterior_density, state_names)


def _compute_decision_state_probability(posterior_density, n_states):
    '''The marginal probability of a state given the posterior_density
    '''
    n_time = len(posterior_density)
    new_shape = (n_time, n_states, -1)
    return np.sum(np.reshape(posterior_density, new_shape), axis=2)


def _compute_max_state(probability, state_names):
    '''The discrete state with the highest probability at the last time
    '''
    end_time_probability = probability[-1, :]
    return (*state_names[np.argmax(end_time_probability)].split('_'),
            np.max(end_time_probability))


def _num_unique_neurons_spiking(spikes):
    '''Number of units that spike per ripple
    '''
    return spikes.sum(axis=0).nonzero()[0].shape[0]


def _num_total_spikes(spikes):
    '''Total number of spikes per ripple
    '''
    return int(spikes.sum(axis=(0, 1)))


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
    return [(session_time_categories
             .loc[ripple_start:ripple_end]
             .value_counts()
             .argmax())
            for ripple_start, ripple_end in ripple_times]


def _subtract_event_related_potential(df):
    return df.apply(lambda x: x - df.mean(axis=1), raw=True)


def is_overlap(band1, band2):
    return (band1[0] < band2[1]) & (band2[0] < band1[1])
