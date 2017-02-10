'''Higher level functions for analyzing the data

'''
from copy import deepcopy
from functools import partial, wraps
from glob import glob
from itertools import combinations
from os.path import abspath, join, pardir
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd

from src.data_processing import (get_area_pair_info,
                                 get_interpolated_position_dataframe,
                                 get_LFP_dataframe,
                                 get_mark_indicator_dataframe,
                                 get_tetrode_pair_info, make_tetrode_dataframe,
                                 reshape_to_segments)
from src.ripple_decoding import (_get_bin_centers, combined_likelihood,
                                 estimate_marked_encoding_model,
                                 estimate_state_transition,
                                 set_initial_conditions, get_ripple_info,
                                 joint_mark_intensity, poisson_mark_likelihood,
                                 predict_state)
from src.spectral import (get_lfps_by_area, multitaper_canonical_coherogram,
                          multitaper_coherogram, power_and_coherence_change)


def coherence_by_ripple_type(epoch_index, animals, ripple_info,
                             ripple_covariate, coherence_name='coherence',
                             multitaper_params={}):
    '''Computes the coherence at each level of a ripple covariate
    from the ripple info dataframe and the differences between those
    levels'''
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    lfps = {index: get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}
    num_lfps = len(lfps)
    num_pairs = int(num_lfps * (num_lfps - 1) / 2)

    grouped = ripple_info.groupby(ripple_covariate)
    params = deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')

    print(
        '\nComputing {coherence_name} for each level of the covariate '
        '"{covariate}"\nfor {num_pairs} pairs of electrodes:'.format(
            coherence_name=coherence_name, covariate=ripple_covariate,
            num_pairs=num_pairs))
    for level_name, ripples_df in grouped:
        ripple_times_by_group = _get_ripple_times(ripples_df)
        print('\tLevel: {level_name} ({num_ripples} ripples)'.format(
            level_name=level_name, num_ripples=len(ripple_times_by_group)))
        reshaped_lfps = {key: reshape_to_segments(
            lfps[key], ripple_times_by_group,
            sampling_frequency=params['sampling_frequency'],
            window_offset=window_of_interest, concat_axis=1)
            for key in lfps}
        for tetrode1, tetrode2 in combinations(
                sorted(reshaped_lfps), 2):
            coherence_df = multitaper_coherogram(
                [reshaped_lfps[tetrode1], reshaped_lfps[tetrode2]],
                **params)
            save_tetrode_pair(coherence_name, ripple_covariate, level_name,
                              tetrode1, tetrode2, coherence_df)
    print('\nComputing the difference in coherence between all levels:')
    for level1, level2 in combinations(
            sorted(grouped.groups.keys()), 2):
        level_difference_name = '{level2}_{level1}'.format(
            level1=level1, level2=level2)
        print('\tLevel Difference: {level2} - {level1}'.format(
            level1=level1, level2=level2))
        for tetrode1, tetrode2 in combinations(
                sorted(reshaped_lfps), 2):
            level1_coherence_df = get_tetrode_pair_from_hdf(
                coherence_name, ripple_covariate, level1,
                tetrode1, tetrode2)
            level2_coherence_df = get_tetrode_pair_from_hdf(
                coherence_name, ripple_covariate, level2,
                tetrode1, tetrode2)
            coherence_difference_df = power_and_coherence_change(
                level1_coherence_df, level2_coherence_df)
            save_tetrode_pair(
                coherence_name, ripple_covariate, level_difference_name,
                tetrode1, tetrode2, coherence_difference_df)
    print('\nSaving Parameters...')
    save_multitaper_parameters(
        epoch_index, coherence_name, multitaper_params)
    save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info)


def canonical_coherence_by_ripple_type(epoch_index, animals, ripple_info,
                                       ripple_covariate,
                                       coherence_name='coherence',
                                       multitaper_params={}):
    '''Computes the canonical coherence at each level of a ripple covariate
    from the ripple info dataframe and the differences between those
    levels'''
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    lfps = {index: get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}

    grouped = ripple_info.groupby(ripple_covariate)
    params = deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')
    print('\nComputing canonical {coherence_name} for each '
          'level of the covariate "{covariate}":'.format(
              coherence_name=coherence_name,
              covariate=ripple_covariate))

    for level_name, ripples_df in grouped:
        ripple_times_by_group = _get_ripple_times(ripples_df)
        print('\tLevel: {level_name} ({num_ripples} ripples)'.format(
            level_name=level_name, num_ripples=len(ripple_times_by_group)))
        reshaped_lfps = {key: reshape_to_segments(
            lfps[key], ripple_times_by_group,
            sampling_frequency=params['sampling_frequency'],
            window_offset=window_of_interest, concat_axis=1).dropna(axis=1)
            for key in lfps}
        area_pairs = combinations(
            sorted(tetrode_info.area.unique()), 2)
        for area1, area2 in area_pairs:
            print('\t\t...{area1} - {area2}'.format(
                area1=area1, area2=area2))
            area1_lfps = get_lfps_by_area(
                area1, tetrode_info, reshaped_lfps)
            area2_lfps = get_lfps_by_area(
                area2, tetrode_info, reshaped_lfps)
            coherogram = multitaper_canonical_coherogram(
                [area1_lfps, area2_lfps], **params)
            save_area_pair(
                coherence_name, ripple_covariate, level_name,
                area1, area2, coherogram, epoch_index)

    print('\nComputing the difference in coherence between all levels:')
    for level1, level2 in combinations(
            sorted(grouped.groups.keys()), 2):
        level_difference_name = '{level2}_{level1}'.format(
            level1=level1, level2=level2)
        print(
            '\tLevel Difference: {level2} - {level1}'.format(
                level1=level1, level2=level2))
        area_pairs = combinations(
            sorted(tetrode_info.area.unique()), 2)
        for area1, area2 in area_pairs:
            print('\t\t...{area1} - {area2}'.format(
                area1=area1, area2=area2))
            level1_coherence_df = get_area_pair_from_hdf(
                coherence_name, ripple_covariate, level1,
                area1, area2, epoch_index)
            level2_coherence_df = get_area_pair_from_hdf(
                coherence_name, ripple_covariate, level2,
                area1, area2, epoch_index)
            coherence_difference_df = power_and_coherence_change(
                level1_coherence_df, level2_coherence_df)
            save_area_pair(
                coherence_name, ripple_covariate, level_difference_name,
                area1, area2, coherence_difference_df, epoch_index)
    print('\nSaving Parameters...')
    save_multitaper_parameters(
        epoch_index, coherence_name, multitaper_params)
    save_area_pair_info(epoch_index, coherence_name, tetrode_info)


def ripple_triggered_coherence(epoch_index, animals, ripple_times,
                               coherence_name='coherence',
                               multitaper_params={}):
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    lfps = {index: get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}
    num_lfps = len(lfps)
    num_pairs = int(num_lfps * (num_lfps - 1) / 2)
    params = deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')

    print('\nComputing ripple-triggered {coherence_name} '
          'for {num_pairs} pairs of electrodes...'.format(
              coherence_name=coherence_name,
              num_pairs=num_pairs))

    reshaped_lfps = {key: reshape_to_segments(
        lfps[key], ripple_times,
        sampling_frequency=params['sampling_frequency'],
        window_offset=window_of_interest,
        concat_axis=1)
        for key in lfps}
    for tetrode1, tetrode2 in combinations(
            sorted(reshaped_lfps), 2):
        coherogram = multitaper_coherogram(
            [reshaped_lfps[tetrode1], reshaped_lfps[tetrode2]], **params)
        coherence_baseline = coherogram.xs(
            coherogram.index.min()[1], level='time')
        coherence_change = power_and_coherence_change(
            coherence_baseline, coherogram)
        save_tetrode_pair(coherence_name, 'all_ripples', 'baseline',
                          tetrode1, tetrode2, coherence_baseline)
        save_tetrode_pair(coherence_name, 'all_ripples', 'ripple_locked',
                          tetrode1, tetrode2, coherogram)
        save_tetrode_pair(coherence_name, 'all_ripples',
                          'ripple_difference_from_baseline',
                          tetrode1, tetrode2, coherence_change)
    save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info)


def ripple_triggered_canonical_coherence(epoch_index, animals,
                                         ripple_times,
                                         coherence_name='coherence',
                                         multitaper_params={}):
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    lfps = {index: get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}
    params = deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')

    reshaped_lfps = {key: reshape_to_segments(
        lfps[key], ripple_times,
        sampling_frequency=params['sampling_frequency'],
        window_offset=window_of_interest,
        concat_axis=1).dropna(axis=1)
        for key in lfps}

    area_pairs = combinations(
        sorted(tetrode_info.area.unique()), 2)
    print('\nComputing ripple-triggered '
          'canonical {coherence_name}:'.format(
              coherence_name=coherence_name))
    for area1, area2 in area_pairs:
        print('\t{area1} - {area2}'.format(area1=area1, area2=area2))
        area1_lfps = get_lfps_by_area(
            area1, tetrode_info, reshaped_lfps)
        area2_lfps = get_lfps_by_area(
            area2, tetrode_info, reshaped_lfps)
        coherogram = multitaper_canonical_coherogram(
            [area1_lfps, area2_lfps], **params)
        coherence_baseline = coherogram.xs(
            coherogram.index.min()[1], level='time')
        coherence_change = power_and_coherence_change(
            coherence_baseline, coherogram)
        save_area_pair(
            coherence_name, 'all_ripples', 'baseline', area1, area2,
            coherence_baseline, epoch_index)
        save_area_pair(
            coherence_name, 'all_ripples', 'ripple_locked', area1, area2,
            coherogram, epoch_index)
        save_area_pair(
            coherence_name, 'all_ripples',
            'ripple_difference_from_baseline', area1, area2,
            coherence_change, epoch_index)
    save_area_pair_info(epoch_index, coherence_name, tetrode_info)


def _get_ripple_times(df):
    '''Retrieves the ripple times from the ripple_info dataframe'''
    return df.loc[
        :, ('ripple_start_time', 'ripple_end_time')].values.tolist()


def save_tetrode_pair(coherence_name, covariate, level, tetrode1,
                      tetrode2, save_df):
    animal, day, epoch = tetrode1[0:3]
    hdf_path = tetrode_pair_hdf_path(
        coherence_name, covariate, level, tetrode1[-1], tetrode2[-1])
    with pd.HDFStore(analysis_file_path(animal, day, epoch)) as store:
        store.put(hdf_path, save_df)


def save_area_pair(coherence_name, covariate, level, area1, area2,
                   save_df, epoch_index):
    animal, day, epoch = epoch_index
    hdf_path = area_pair_hdf_path(
        coherence_name, covariate, level, area1, area2)
    with pd.HDFStore(analysis_file_path(animal, day, epoch)) as store:
        store.put(hdf_path, save_df)


def get_tetrode_pair_from_hdf(coherence_name, covariate, level,
                              tetrode1, tetrode2):
    animal, day, epoch = tetrode1[0:3]
    hdf_path = tetrode_pair_hdf_path(
        coherence_name, covariate, level, tetrode1[-1], tetrode2[-1])
    try:
        return pd.read_hdf(
            analysis_file_path(animal, day, epoch), key=hdf_path)
    except KeyError:
        print('Could not load tetrode pair:'
              'animal={animal}, day={day}, epoch={epoch}'
              'tetrode {tetrode1} - tetrode {tetrode2}\n'.format(
                animal=animal, day=day, epoch=epoch, tetrode1=tetrode1,
                tetrode2=tetrode2
              ))


def get_area_pair_from_hdf(coherence_name, covariate, level, area1, area2,
                           epoch_index):
    animal, day, epoch = epoch_index
    hdf_path = area_pair_hdf_path(
        coherence_name, covariate, level, area1, area2)
    return pd.read_hdf(
        analysis_file_path(animal, day, epoch), key=hdf_path)


def tetrode_pair_hdf_path(coherence_name, covariate, level,
                          tetrode1, tetrode2):
    return '/{coherence_name}/tetrode{tetrode1:04d}_tetrode{tetrode2:04d}/{covariate}/{level}'.format(
        coherence_name=coherence_name, covariate=covariate,
        level=level, tetrode1=tetrode1, tetrode2=tetrode2)


def area_pair_hdf_path(coherence_name, covariate, level, area1, area2):
    return '/{coherence_name}/{area1}_{area2}/{covariate}/{level}'.format(
        coherence_name=coherence_name, covariate=covariate,
        level=level, area1=area1, area2=area2)


def analysis_file_path(animal, day, epoch):
    filename = '{animal}_{day:02d}_{epoch:02d}.h5'.format(
        animal=animal, day=day, epoch=epoch)
    return join(
        abspath(pardir), 'Processed-Data', filename)


def save_multitaper_parameters(epoch_index, coherence_name,
                               multitaper_params):
    coherence_node_name = '/{coherence_name}'.format(
        coherence_name=coherence_name)
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        store.get_node(
            coherence_node_name)._v_attrs.multitaper_parameters = \
            multitaper_params


def save_ripple_info(epoch_index, ripple_info):
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        store.put('/ripple_info', ripple_info)


def save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info):
    hdf_path = '/{coherence_name}/tetrode_info'.format(
        coherence_name=coherence_name)
    hdf_pair_path = '/{coherence_name}/tetrode_pair_info'.format(
        coherence_name=coherence_name)
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        with catch_warnings():
            simplefilter('ignore')
            store.put(hdf_path, tetrode_info)
            store.put(hdf_pair_path,
                      get_tetrode_pair_info(tetrode_info))


def save_area_pair_info(epoch_index, coherence_name, tetrode_info):
    hdf_pair_path = '/{coherence_name}/area_pair_info'.format(
        coherence_name=coherence_name)
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        with catch_warnings():
            simplefilter('ignore')
            store.put(
                hdf_pair_path, get_area_pair_info(
                    tetrode_info, epoch_index))


def get_tetrode_pair_group_from_hdf(tetrode_pair_index, coherence_name,
                                    covariate, level):
    '''Given a list of tetrode indices and specifiers for the path,
    returns a panel object of the corresponding coherence dataframes'''
    return pd.Panel({(tetrode1, tetrode2): get_tetrode_pair_from_hdf(
        coherence_name, covariate, level, tetrode1, tetrode2)
        for tetrode1, tetrode2 in tetrode_pair_index})


def get_all_tetrode_pair_info(coherence_name):
    '''Retrieves all the hdf5 files from the Processed Data directory and
    returns the tetrode pair info dataframe'''
    file_path = join(abspath(
        pardir), 'Processed-Data', '*.h5')
    hdf5_files = glob(file_path)
    hdf_path = '/{coherence_name}/tetrode_pair_info'.format(
        coherence_name=coherence_name)
    return pd.concat(
        [pd.read_hdf(filename, key=hdf_path)
         for filename in hdf5_files]).sort_index()


def get_all_tetrode_info(coherence_name):
    '''Retrieves all the hdf5 files from the Processed Data directory
    and returns the tetrode pair info dataframe'''
    file_path = join(abspath(
        pardir), 'Processed-Data', '*.h5')
    hdf5_files = glob(file_path)
    hdf_path = '/{coherence_name}/tetrode_info'.format(
        coherence_name=coherence_name)
    return pd.concat(
        [pd.read_hdf(filename, key=hdf_path)
         for filename in hdf5_files]).sort_index()


def find_power_spectrum_from_pair_index(target_tetrode, coherence_name,
                                        covariate, level,
                                        tetrode_pair_info):
    tetrode1, tetrode2 = next(
        (tetrode1, tetrode2) for tetrode1, tetrode2
        in tetrode_pair_info.index
        if tetrode1 == target_tetrode or tetrode2 == target_tetrode)
    coherence_df = get_tetrode_pair_from_hdf(
        coherence_name, covariate, level, tetrode1, tetrode2)
    power_spectrum_name = 'power_spectrum{}'.format(
        (tetrode1, tetrode2).index(target_tetrode) + 1)
    return pd.DataFrame(coherence_df[power_spectrum_name].rename('power'))


def merge_symmetric_key_pairs(pair_dict):
    '''If a 2-element key is the same except for the order, merge the
    Pandas index corresponding to the key.

    For example, a dictionary with keys ('a', 'b') and ('b', 'a') will be
    combined into a key ('a', 'b')

    Parameters
    ----------
    pair_dict : dict
        A dictionary with keys that are 2-element tuples and values that
        are a Pandas index.

    Returns
    -------
    merged_dict : dict

    Examples
    --------
    >>> import pandas as pd
    >>> test_dict = {('a', 'a'): pd.Index([1, 2, 3]),
                     ('a', 'b'): pd.Index([4, 5, 6]),
                     ('b', 'a'): pd.Index([7, 8, 9]),
                     ('b', 'c'): pd.Index([10, 11, 12])}
    >>> merge_symmetric_key_pairs(test_dict)
    {('a', 'a'): Int64Index([1, 2, 3], dtype='int64'),
     ('a', 'b'): Int64Index([4, 5, 6, 7, 8, 9], dtype='int64'),
     ('b', 'c'): Int64Index([10, 11, 12], dtype='int64')}

    '''
    skip_list = list()
    merged_dict = dict()

    for area1, area2 in sorted(pair_dict):
        if area1 == area2:
            merged_dict[(area1, area2)] = pair_dict[(area1, area2)]
        elif ((area2, area1) in pair_dict and
              (area1, area2) not in skip_list):
            skip_list.append((area2, area1))
            merged_dict[(area1, area2)] = pair_dict[
                (area1, area2)].union(pair_dict[(area2, area1)])
        elif (area1, area2) not in skip_list:
            merged_dict[(area1, area2)] = pair_dict[(area1, area2)]

    return merged_dict


def decode_ripple_clusterless(epoch_index, animals, ripple_times,
                              sampling_frequency=1500,
                              n_place_bins=61,
                              place_std_deviation=None,
                              mark_std_deviation=20):
    # Encode
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_index]

    mark_variables = ['channel_1_max', 'channel_2_max', 'channel_3_max',
                      'channel_4_max']
    tetrode_marks = [(get_mark_indicator_dataframe(tetrode_index, animals)
                      .loc[:, mark_variables])
                     for tetrode_index in tetrode_info.loc[
        tetrode_info.area.isin(['CA1', 'iCA1']) &
        (tetrode_info.descrip != 'CA1Ref'), :]]

    position_variables = ['linear_distance', 'trajectory_direction',
                          'speed']
    position_info = (get_interpolated_position_dataframe(
        epoch_index, animals).loc[:, position_variables])

    train_position_info = position_info.query('speed > 4')

    place = _get_place(train_position_info)
    place_at_spike = [_get_place_at_spike(mark_tetrode_data,
                                          train_position_info)
                      for mark_tetrode_data in tetrode_marks]
    training_marks = [_get_training_marks(mark_tetrode_data,
                                          train_position_info,
                                          mark_variables)
                      for mark_tetrode_data in tetrode_marks]

    place_bin_edges = np.linspace(
        np.floor(position_info.linear_distance.min()),
        np.ceil(position_info.linear_distance.max()),
        n_place_bins + 1)
    place_bin_centers = _get_bin_centers(place_bin_edges)

    if place_std_deviation is None:
        place_std_deviation = place_bin_edges[1] - place_bin_edges[0]

    (place_occupancy, ground_process_intensity, place_field_estimator,
     training_marks) = estimate_marked_encoding_model(
        place_bin_centers, place, place_at_spike, training_marks,
        place_std_deviation=place_std_deviation)

    fixed_joint_mark_intensity = partial(
        joint_mark_intensity, place_field_estimator=place_field_estimator,
        place_occupancy=place_occupancy, training_marks=training_marks,
        mark_std_deviation=mark_std_deviation)

    combined_likelihood_kwargs = dict(
        likelihood_function=poisson_mark_likelihood,
        likelihood_kwargs=dict(
            joint_mark_intensity=fixed_joint_mark_intensity,
            ground_process_intensity=ground_process_intensity)
    )

    # Fit state transition model
    print('\tFitting state transition model...')
    state_transition = estimate_state_transition(
        train_position_info, place_bin_edges)

    # Initial Conditions
    print('\tSetting initial conditions...')
    state_names = ['outbound_forward', 'outbound_reverse',
                   'inbound_forward', 'inbound_reverse']
    n_states = len(state_names)
    initial_conditions = set_initial_conditions(
        place_bin_edges, place_bin_centers, n_states)

    # Decode
    decoder_kwargs = dict(
        initial_conditions=initial_conditions,
        state_transition=state_transition,
        likelihood_function=combined_likelihood,
        likelihood_kwargs=combined_likelihood_kwargs
    )

    test_marks = _get_ripple_marks(
        tetrode_marks, ripple_times, sampling_frequency)

    posterior_density = [predict_state(ripple_marks, **decoder_kwargs)
                         for ripple_marks in test_marks]
    test_spikes = [np.mean(~np.isnan(marks), axis=2)
                   for marks in test_marks]

    return get_ripple_info(
        posterior_density, test_spikes, ripple_times,
        state_names, position_info.index)


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
def _get_place_at_spike(mark_tetrode_data, train_position_info,
                        place_measure='linear_distance'):
    return {trajectory_direction: (grouped.dropna()
                                   .loc[:, place_measure].values)
            for trajectory_direction, grouped
            in (mark_tetrode_data
                .join(train_position_info)
                .groupby('trajectory_direction'))}


@_convert_to_states
def _get_training_marks(mark_tetrode_data, train_position_info,
                        mark_variables):
    return {trajectory_direction: (grouped.dropna()
                                   .loc[:, mark_variables].values)
            for trajectory_direction, grouped
            in (mark_tetrode_data
                .join(train_position_info)
                .groupby('trajectory_direction'))}


def _get_ripple_marks(tetrode_marks, ripple_times, sampling_frequency):
    mark_ripples = [reshape_to_segments(
        mark_tetrode_data, ripple_times,
        concat_axis=0, sampling_frequency=sampling_frequency)
        for mark_tetrode_data in tetrode_marks]

    return [np.stack([df.loc[ripple_ind + 1, :].values
                      for df in mark_ripples], axis=1)
            for ripple_ind in np.arange(len(ripple_times))]
