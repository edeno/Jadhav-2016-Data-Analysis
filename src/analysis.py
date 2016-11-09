import itertools
import data_processing
import ripple_detection
import ripple_decoding
import spectral


def coherence_by_ripple_type(epoch_index, animals, ripple_covariate, coherence_name,
                             ripple_detection_function=ripple_detection.Kay_method,
                             multitaper_params={}, decoding_params={}):
    tetrode_info = data_processing.make_tetrode_dataframe(animals)[epoch_index]
    ripple_times = ripple_detection.get_epoch_ripples(
        epoch_index, animals, sampling_frequency=multitaper_params['sampling_frequency'],
        ripple_detection_function=ripple_detection_function)
    ripple_info = ripple_decoding.decode_ripple(
        epoch_index, animals, ripple_times, **decoding_params)[0]
    lfps = {index: data_processing.get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}

    grouped = ripple_info.groupby(ripple_covariate)
    window_of_interest = multitaper_params.pop('window_of_interest')

    for level_name, ripples_df in grouped:
        ripple_times_by_group = _get_ripple_times(ripples_df)
        reshaped_lfps = {key: data_processing.reshape_to_segments(
            lfps[key], ripple_times_by_group, window_offset=window_of_interest, concat_axis=1)
                         for key in lfps}
        for tetrode1, tetrode2 in itertools.combinations(sorted(reshaped_lfps), 2):
            coherence_df = spectral.multitaper_coherogram(
                [reshaped_lfps[tetrode1], reshaped_lfps[tetrode2]], **multitaper_params)
            save_tetrode_pair(coherence_name, ripple_covariate, level_name,
                              tetrode1, tetrode2, coherence_df)

    for level1, level2 in itertools.combinations(grouped.groups.keys(), 2):
        level_difference_name = '{level2}_{level1}'.format(level1=level1, level2=level2)
        for tetrode1, tetrode2 in itertools.combinations(sorted(reshaped_lfps), 2):
            level1_coherence_df = get_tetrode_pair_from_hdf(
                coherence_name, ripple_covariate, level1, tetrode1, tetrode2)
            level2_coherence_df = get_tetrode_pair_from_hdf(
                coherence_name, ripple_covariate, level2, tetrode1, tetrode2)
            coherence_difference_df = spectral.power_and_coherence_change(
                level1_coherence_df, level2_coherence_df)
            save_tetrode_pair(coherence_name, ripple_covariate, level_difference_name,
                              tetrode1, tetrode2, coherence_difference_df)

    multitaper_params['window_of_interest'] = window_of_interest
    save_multitaper_parameters(epoch_index, coherence_name, multitaper_params)
    save_ripple_info(epoch_index, ripple_info)
    save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info)


def _get_ripple_times(df):
    return df.loc[:, ('ripple_start_time', 'ripple_end_time')].values.tolist()


def save_tetrode_pair(coherence_name, covariate, level, tetrode1, tetrode2, save_df):
    animal, day, epoch = tetrode1[0:3]
    hdf_path = tetrode_pair_hdf_path(coherence_name, covariate, level, tetrode1[-1], tetrode2[-1])
    with pd.HDFStore(analysis_file_path(animal, day, epoch)) as store:
        store.put(hdf_path, save_df)


def get_tetrode_pair_from_hdf(coherence_name, covariate, level, tetrode1, tetrode2):
    animal, day, epoch = tetrode1[0:3]
    hdf_path = tetrode_pair_hdf_path(coherence_name, covariate, level, tetrode1[-1], tetrode2[-1])
    return pd.read_hdf(analysis_file_path(animal, day, epoch), key=hdf_path)


def tetrode_pair_hdf_path(coherence_name, covariate, level, tetrode1, tetrode2):
    return '{coherence_name}/tetrode{tetrode1}_tetrode{tetrode2}/{covariate}/{level}'.format(
        coherence_name=coherence_name, covariate=covariate,
        level=level, tetrode1=tetrode1, tetrode2=tetrode2)


def analysis_file_path(animal, day, epoch):
    filename = '{animal}_{day}_{epoch}.h5'.format(animal=animal, day=day, epoch=epoch)
    return os.path.join(os.path.abspath(os.path.pardir), 'Processed-Data', filename)


def save_multitaper_parameters(epoch_index, coherence_name, multitaper_params):
    coherence_node_name = '/{coherence_name}'.format(coherence_name=coherence_name)
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        store.get_node(coherence_node_name)._v_attrs.multitaper_parameters = multitaper_params


def save_ripple_info(epoch_index, ripple_info):
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        store.put('/ripple_info', ripple_info)


def save_tetrode_pair_info(epoch_index, coherence_name, tetrode_info):
    hdf_path = '{coherence_name}/tetrode_pair_info'.format(coherence_name=coherence_name)
    with pd.HDFStore(analysis_file_path(*epoch_index)) as store:
        store.put(hdf_path, data_processing.get_tetrode_pair_info(tetrode_info))


