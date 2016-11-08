import itertools
import data_processing
import ripple_detection
import ripple_decoding
import spectral


def coherence_by_ripple_type(epoch_index, animals, ripple_covariate,
                             ripple_detection_function=ripple_detection.Kay_method,
                             multitaper_params={}, decoding_params={}, sampling_frequency=1500):
    tetrode_info = data_processing.make_tetrode_dataframe(animals)[epoch_index]
    ripple_times = ripple_detection.get_epoch_ripples(
        epoch_index, animals, sampling_frequency=sampling_frequency,
        ripple_detection_function=ripple_detection_function)
    ripple_info, _, _, _, _ = ripple_decoding.decode_ripple(epoch_index, animals,
                                                            ripple_times, **decoding_params)
    lfps = {index: data_processing.get_LFP_dataframe(index, animals)
            for index in tetrode_info.index}

    grouped = ripple_info.groupby(ripple_covariate)
    window_of_interest = multitaper_params.pop('window_of_interest')
    coherence_by_ripple_type = {ripple_type: _get_coherence_for_all_pairs(lfps, ripples,
                                                                          multitaper_params,
                                                                          window_of_interest)
                                for ripple_type, ripples in grouped}

    difference_of_levels = itertools.combinations(grouped.groups.keys(), 2)
    for level1, level2 in difference_of_levels:
        group_name = '{level2} - {level1}'.format(level1=level1, level2=level2)
        coherence_by_ripple_type[group_name] = {
            tetrode_pair: spectral.power_and_coherence_change(
                coherence_by_ripple_type[level1][tetrode_pair],
                coherence_by_ripple_type[level2][tetrode_pair])
            for tetrode_pair in coherence_by_ripple_type[level1]}

    return coherence_by_ripple_type


def _get_coherence_for_all_pairs(lfps, ripples_df, multitaper_params, window_of_interest):
    ripple_times = _get_ripple_times(ripples_df)
    reshaped_lfps = {key: data_processing.reshape_to_segments(
        lfps[key], ripple_times, window_offset=window_of_interest, concat_axis=1)
                     for key in lfps}
    return {(lfp1, lfp2): spectral.multitaper_coherogram(
        [reshaped_lfps[lfp1], reshaped_lfps[lfp2]], **multitaper_params)
            for lfp1, lfp2 in itertools.combinations(reshaped_lfps.keys(), 2)}


def _get_ripple_times(df):
    return df.loc[:, ('ripple_start_time', 'ripple_end_time')].values.tolist()
