import copy
import os
import glob
import warnings
import itertools
import pandas as pd
import data_processing
import spectral


def coherence_by_ripple_type(epoch_index, animals, ripple_info, ripple_covariate,
                             coherence_name='coherence', multitaper_params={}):
    '''Computes the coherence at each level of a ripple covariate
    from the ripple info dataframe and the differences between those levels'''
    tetrode_info = data_processing.make_tetrode_dataframe(animals)[epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    print(tetrode_info.loc[:, ['area', 'depth', 'descrip']])
    lfps = {index: data_processing.get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}
    num_lfps = len(lfps)
    num_pairs = int(num_lfps * (num_lfps - 1) / 2)

    grouped = ripple_info.groupby(ripple_covariate)
    params = copy.deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')

    print('\nComputing {coherence_name} for each level of the covariate "{covariate}"'
          '\nfor {num_pairs} pairs of electrodes:'.format(coherence_name=coherence_name,
                                                          covariate=ripple_covariate,
                                                          num_pairs=num_pairs))
    for level_name, ripples_df in grouped:
        ripple_times_by_group = _get_ripple_times(ripples_df)
        print('\tLevel: {level_name} ({num_ripples} ripples)'.format(
            level_name=level_name, num_ripples=len(ripple_times_by_group)))
        reshaped_lfps = {key: data_processing.reshape_to_segments(
            lfps[key], ripple_times_by_group,
            sampling_frequency=params['sampling_frequency'],
            window_offset=window_of_interest, concat_axis=1)
            for key in lfps}
        for tetrode1, tetrode2 in itertools.combinations(sorted(reshaped_lfps), 2):
            coherence_df = spectral.multitaper_coherogram(
                [reshaped_lfps[tetrode1], reshaped_lfps[tetrode2]], **params)
            save_tetrode_pair(coherence_name, ripple_covariate, level_name,
                              tetrode1, tetrode2, coherence_df)
    print('\nComputing the difference in coherence between all levels:')
    for level1, level2 in itertools.combinations(sorted(grouped.groups.keys()), 2):
        level_difference_name = '{level2}_{level1}'.format(
            level1=level1, level2=level2)
        print(
            '\tLevel Difference: {level2} - {level1}'.format(level1=level1, level2=level2))
        for tetrode1, tetrode2 in itertools.combinations(sorted(reshaped_lfps), 2):
            level1_coherence_df = get_tetrode_pair_from_hdf(
                coherence_name, ripple_covariate, level1, tetrode1, tetrode2)
            level2_coherence_df = get_tetrode_pair_from_hdf(
                coherence_name, ripple_covariate, level2, tetrode1, tetrode2)
            coherence_difference_df = spectral.power_and_coherence_change(
                level1_coherence_df, level2_coherence_df)
            save_tetrode_pair(coherence_name, ripple_covariate, level_difference_name,
                              tetrode1, tetrode2, coherence_difference_df)
    print('\nSaving Parameters...')
    save_multitaper_parameters(epoch_index, coherence_name, multitaper_params)
    save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info)


def canonical_coherence_by_ripple_type(epoch_index, animals, ripple_info, ripple_covariate,
                                       coherence_name='coherence', multitaper_params={}):
    '''Computes the canonical coherence at each level of a ripple covariate
    from the ripple info dataframe and the differences between those levels'''
    tetrode_info = data_processing.make_tetrode_dataframe(animals)[epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    print(tetrode_info.loc[:, ['area', 'depth', 'descrip']])
    lfps = {index: data_processing.get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}

    grouped = ripple_info.groupby(ripple_covariate)
    params = copy.deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')
    print('\nComputing canonical {coherence_name} for each '
          'level of the covariate "{covariate}":'.format(
              coherence_name=coherence_name,
              covariate=ripple_covariate))

    for level_name, ripples_df in grouped:
        ripple_times_by_group = _get_ripple_times(ripples_df)
        print('\tLevel: {level_name} ({num_ripples} ripples)'.format(
            level_name=level_name, num_ripples=len(ripple_times_by_group)))
        reshaped_lfps = {key: data_processing.reshape_to_segments(
            lfps[key], ripple_times_by_group,
            sampling_frequency=params['sampling_frequency'],
            window_offset=window_of_interest, concat_axis=1).dropna(axis=1)
            for key in lfps}
        area_pairs = itertools.combinations(
            sorted(tetrode_info.area.unique()), 2)
        for area1, area2 in area_pairs:
            print('\t\t...{area1} - {area2}'.format(area1=area1, area2=area2))
            area1_lfps = spectral.get_lfps_by_area(
                area1, tetrode_info, reshaped_lfps)
            area2_lfps = spectral.get_lfps_by_area(
                area2, tetrode_info, reshaped_lfps)
            coherogram = spectral.multitaper_canonical_coherogram(
                [area1_lfps, area2_lfps], **params)
            save_area_pair(
                coherence_name, ripple_covariate, level_name,
                area1, area2, coherogram, epoch_index)

    print('\nComputing the difference in coherence between all levels:')
    for level1, level2 in itertools.combinations(sorted(grouped.groups.keys()), 2):
        level_difference_name = '{level2}_{level1}'.format(
            level1=level1, level2=level2)
        print(
            '\tLevel Difference: {level2} - {level1}'.format(level1=level1, level2=level2))
        area_pairs = itertools.combinations(
            sorted(tetrode_info.area.unique()), 2)
        for area1, area2 in area_pairs:
            print('\t\t...{area1} - {area2}'.format(area1=area1, area2=area2))
            level1_coherence_df = get_area_pair_from_hdf(
                coherence_name, ripple_covariate, level1, area1, area2, epoch_index)
            level2_coherence_df = get_area_pair_from_hdf(
                coherence_name, ripple_covariate, level2, area1, area2, epoch_index)
            coherence_difference_df = spectral.power_and_coherence_change(
                level1_coherence_df, level2_coherence_df)
            save_area_pair(coherence_name, ripple_covariate, level_difference_name,
                           area1, area2, coherence_difference_df, epoch_index)
    print('\nSaving Parameters...')
    save_multitaper_parameters(epoch_index, coherence_name, multitaper_params)
    save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info)


def ripple_triggered_coherence(epoch_index, animals, ripple_times,
                               coherence_name='coherence', multitaper_params={}):
    tetrode_info = data_processing.make_tetrode_dataframe(animals)[epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    print(tetrode_info.loc[:, ['area', 'depth', 'descrip']])
    lfps = {index: data_processing.get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}
    num_lfps = len(lfps)
    num_pairs = int(num_lfps * (num_lfps - 1) / 2)
    params = copy.deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')

    print('\nComputing ripple-triggered {coherence_name} '
          'for {num_pairs} pairs of electrodes...'.format(
              coherence_name=coherence_name,
              num_pairs=num_pairs))

    reshaped_lfps = {key: data_processing.reshape_to_segments(
        lfps[key], ripple_times,
        sampling_frequency=params['sampling_frequency'],
        window_offset=window_of_interest,
        concat_axis=1)
        for key in lfps}
    for tetrode1, tetrode2 in itertools.combinations(sorted(reshaped_lfps), 2):
        coherogram = spectral.multitaper_coherogram(
            [reshaped_lfps[tetrode1], reshaped_lfps[tetrode2]], **params)
        coherence_baseline = coherogram.xs(
            coherogram.index.min()[1], level='time')
        coherence_change = spectral.power_and_coherence_change(
            coherence_baseline, coherogram)
        save_tetrode_pair(coherence_name, 'all_ripples', '',
                          tetrode1, tetrode2, coherence_change)


def ripple_triggered_canonical_coherence(epoch_index, animals, ripple_times,
                                         coherence_name='coherence', multitaper_params={}):
    tetrode_info = data_processing.make_tetrode_dataframe(animals)[epoch_index]
    tetrode_info = tetrode_info[
        ~tetrode_info.descrip.str.endswith('Ref').fillna(False)]
    print(tetrode_info.loc[:, ['area', 'depth', 'descrip']])
    print('Number of tetrodes per area:')
    print(tetrode_info.area.value_counts())
    lfps = {index: data_processing.get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}
    params = copy.deepcopy(multitaper_params)
    window_of_interest = params.pop('window_of_interest')

    reshaped_lfps = {key: data_processing.reshape_to_segments(
        lfps[key], ripple_times,
        sampling_frequency=params['sampling_frequency'],
        window_offset=window_of_interest,
        concat_axis=1).dropna(axis=1)
        for key in lfps}

    area_pairs = itertools.combinations(sorted(tetrode_info.area.unique()), 2)
    print('\nComputing ripple-triggered '
          'canonical {coherence_name}:'.format(coherence_name=coherence_name))
    for area1, area2 in area_pairs:
        print('\t{area1} - {area2}'.format(area1=area1, area2=area2))
        area1_lfps = spectral.get_lfps_by_area(
            area1, tetrode_info, reshaped_lfps)
        area2_lfps = spectral.get_lfps_by_area(
            area2, tetrode_info, reshaped_lfps)
        coherogram = spectral.multitaper_canonical_coherogram(
            [area1_lfps, area2_lfps], **params)
        coherence_baseline = coherogram.xs(
            coherogram.index.min()[1], level='time')
        coherence_change = spectral.power_and_coherence_change(
            coherence_baseline, coherogram)
        save_area_pair(
            coherence_name, 'all_ripples', '', area1, area2, coherence_change, epoch_index)


def _get_ripple_times(df):
    '''Retrieves the ripple times from the ripple_info dataframe'''
    return df.loc[:, ('ripple_start_time', 'ripple_end_time')].values.tolist()


def save_tetrode_pair(coherence_name, covariate, level, tetrode1, tetrode2, save_df):
    animal, day, epoch = tetrode1[0:3]
    hdf_path = tetrode_pair_hdf_path(
        coherence_name, covariate, level, tetrode1[-1], tetrode2[-1])
    with pd.HDFStore(analysis_file_path(animal, day, epoch)) as store:
        store.put(hdf_path, save_df)


def save_area_pair(coherence_name, covariate, level, area1, area2, save_df, epoch_index):
    animal, day, epoch = epoch_index
    hdf_path = area_pair_hdf_path(
        coherence_name, covariate, level, area1, area2)
    with pd.HDFStore(analysis_file_path(animal, day, epoch)) as store:
        store.put(hdf_path, save_df)


def get_tetrode_pair_from_hdf(coherence_name, covariate, level, tetrode1, tetrode2):
    animal, day, epoch = tetrode1[0:3]
    hdf_path = tetrode_pair_hdf_path(
        coherence_name, covariate, level, tetrode1[-1], tetrode2[-1])
    return pd.read_hdf(analysis_file_path(animal, day, epoch), key=hdf_path)


def get_area_pair_from_hdf(coherence_name, covariate, level, area1, area2, epoch_index):
    animal, day, epoch = epoch_index
    hdf_path = area_pair_hdf_path(
        coherence_name, covariate, level, area1, area2)
    return pd.read_hdf(analysis_file_path(animal, day, epoch), key=hdf_path)


def tetrode_pair_hdf_path(coherence_name, covariate, level, tetrode1, tetrode2):
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
    return os.path.join(os.path.abspath(os.path.pardir), 'Processed-Data', filename)


def save_multitaper_parameters(epoch_index, coherence_name, multitaper_params):
    coherence_node_name = '/{coherence_name}'.format(
        coherence_name=coherence_name)
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        store.get_node(
            coherence_node_name)._v_attrs.multitaper_parameters = multitaper_params


def save_ripple_info(epoch_index, ripple_info):
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        store.put('/ripple_info', ripple_info)


def save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info):
    hdf_path = '/{coherence_name}/tetrode_info'.format(
        coherence_name=coherence_name)
    hdf_pair_path = '/{coherence_name}/tetrode_pair_info'.format(
        coherence_name=coherence_name)
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            store.put(hdf_path, tetrode_info)
            store.put(hdf_pair_path,
                      data_processing.get_tetrode_pair_info(tetrode_info))


def get_tetrode_pair_group_from_hdf(tetrode_pair_index, coherence_name, covariate, level):
    '''Given a list of tetrode indices and specifiers for the path,
    returns a panel object of the corresponding coherence dataframes'''
    return pd.Panel({(tetrode1, tetrode2): get_tetrode_pair_from_hdf(
        coherence_name, covariate, level, tetrode1, tetrode2)
        for tetrode1, tetrode2 in tetrode_pair_index})


def get_all_tetrode_pair_info(coherence_name):
    '''Retrieves all the hdf5 files from the Processed Data directory and returns the tetrode
    pair info dataframe'''
    file_path = os.path.join(os.path.abspath(
        os.path.pardir), 'Processed-Data', '*.h5')
    hdf5_files = glob.glob(file_path)
    hdf_path = '/{coherence_name}/tetrode_pair_info'.format(
        coherence_name=coherence_name)
    return pd.concat([pd.read_hdf(filename, key=hdf_path) for filename in hdf5_files]).sort_index()


def get_all_tetrode_info(coherence_name):
    '''Retrieves all the hdf5 files from the Processed Data directory and returns the tetrode
    pair info dataframe'''
    file_path = os.path.join(os.path.abspath(
        os.path.pardir), 'Processed-Data', '*.h5')
    hdf5_files = glob.glob(file_path)
    hdf_path = '/{coherence_name}/tetrode_info'.format(
        coherence_name=coherence_name)
    return pd.concat([pd.read_hdf(filename, key=hdf_path) for filename in hdf5_files]).sort_index()


def find_power_spectrum_from_pair_index(target_tetrode, coherence_name,
                                        covariate, level, tetrode_pair_info):
    tetrode1, tetrode2 = next((tetrode1, tetrode2)
                              for tetrode1, tetrode2 in tetrode_pair_info.index
                              if tetrode1 == target_tetrode or tetrode2 == target_tetrode)
    coherence_df = get_tetrode_pair_from_hdf(
        coherence_name, covariate, level, tetrode1, tetrode2)
    power_spectrum_name = 'power_spectrum{}'.format(
        (tetrode1, tetrode2).index(target_tetrode) + 1)
    return pd.DataFrame(coherence_df[power_spectrum_name].rename('power'))
