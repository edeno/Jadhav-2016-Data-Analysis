import itertools
import pandas as pd
import data_processing
import ripple_detection
import ripple_decoding
import spectral


def coherence_by_ripple_type(epoch_index, animals,
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
    coherence_by_ripple_type = {ripple_type: _get_coherence_for_all_pairs(lfps, ripples,
                                                                          multitaper_params)
                                for ripple_type, ripples in ripple_info.groupby('ripple_type')}
    return coherence_by_ripple_type, get_tetrode_pair_info(tetrode_info)


def _get_coherence_for_all_pairs(lfps, ripples, multitaper_params):
    return {(lfp1, lfp2): spectral.difference_from_baseline_coherence(
                [lfps[lfp1], lfps[lfp2]], _get_ripple_times(ripples), **multitaper_params)
            for lfp1, lfp2 in itertools.combinations(lfps.keys(), 2)}


def _get_ripple_times(df):
    return df.loc[:, ('ripple_start_time', 'ripple_end_time')].values.tolist()


def get_tetrode_pair_info(tetrode_info):
    pair_index = pd.MultiIndex.from_tuples(list(itertools.combinations(tetrode_info.index, 2)),
                                           names=['tetrode1', 'tetrode2'])
    no_rename = {'animal_1': 'animal',
                 'day_1': 'day',
                 'epoch_ind_1': 'epoch_ind'}
    tetrode1 = (tetrode_info
                .loc[pair_index.get_level_values('tetrode1')]
                .reset_index(drop=True)
                .add_suffix('_1')
                .rename(columns=no_rename)
                )
    tetrode2 = (tetrode_info
                .loc[pair_index.get_level_values('tetrode2')]
                .reset_index(drop=True)
                .drop(['animal', 'day', 'epoch_ind'], axis=1)
                .add_suffix('_2')
                )
    return (pd.concat([tetrode1, tetrode2], axis=1)
            .set_index(pair_index))
