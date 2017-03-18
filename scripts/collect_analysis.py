from itertools import combinations
from sys import exit

import pandas as pd

from src.data_processing import (get_all_area_pair_info,
                                 get_all_ripple_info,
                                 get_all_tetrode_info,
                                 get_all_tetrode_pair_info,
                                 get_brain_area_pairs_canonical_coherence,
                                 get_brain_area_pairs_coherence,
                                 get_brain_area_pairs_group_delay,
                                 get_brain_area_power)
from src.parameters import (FREQUENCY_BANDS, MULTITAPER_PARAMETERS,
                            RIPPLE_COVARIATES)

RESULTS_FILE = 'results.h5'


def main():
    tetrode_info = get_all_tetrode_info()
    tetrode_pair_info = get_all_tetrode_pair_info()
    area_pair_info = get_all_area_pair_info()
    ripple_info = get_all_ripple_info()
    save_analysis(tetrode_info, tetrode_pair_info, area_pair_info,
                  ripple_info)

    animal_groups = zip(tetrode_info.groupby('animal'),
                        tetrode_pair_info.groupby('animal'),
                        area_pair_info.groupby('animal'))

    for (animal_tetrode_info, animal_tetrode_pair_info,
         animal_area_pair_info) in animal_groups:
        save_analysis(animal_tetrode_info[1], animal_tetrode_pair_info[1],
                      animal_area_pair_info[1], ripple_info,
                      label=animal_tetrode_info[0])

    epoch_groups = zip(tetrode_info.groupby('epoch'),
                       tetrode_pair_info.groupby('epoch'),
                       area_pair_info.groupby('epoch'))

    for (epoch_tetrode_info, epoch_tetrode_pair_info,
         epoch_area_pair_info) in epoch_groups:
        save_analysis(epoch_tetrode_info[1], epoch_tetrode_pair_info[1],
                      epoch_area_pair_info[1], ripple_info,
                      label=epoch_tetrode_info[0])

    with pd.HDFStore(RESULTS_FILE) as store:
        store.put('/ripple_info', ripple_info)
        store.put('/tetrode_info', tetrode_info)
        store.put('/tetrode_pair_info', tetrode_pair_info)
        store.put('/area_pair_info', area_pair_info)


def save(analysis_function, path, args=None):
    with pd.HDFStore(RESULTS_FILE) as store:
        store.put(path, analysis_function(*args))


def save_mean(label, multitaper_parameter_name, covariate, level,
              tetrode_info, tetrode_pair_info, area_pair_info):
    path = '/'.join([label, multitaper_parameter_name, 'power', covariate,
                     level])
    save(get_brain_area_power, path, args=(
        multitaper_parameter_name, covariate, level,
        tetrode_info))

    path = '/'.join([label, multitaper_parameter_name, 'coherence',
                     covariate, level])
    save(get_brain_area_pairs_coherence, path, args=(
        multitaper_parameter_name, covariate, level,
        tetrode_pair_info))

    path = '/'.join([label, multitaper_parameter_name,
                     'canonical_coherence', covariate, level])
    save(get_brain_area_pairs_canonical_coherence, path, args=(
        multitaper_parameter_name, covariate, level,
        area_pair_info))

    for frequency_band in FREQUENCY_BANDS:
        if level not in ['baseline', 'ripple_difference_from_baseline']:
            path = '/'.join([label, multitaper_parameter_name,
                             'group_delay', covariate, level,
                             frequency_band])
            save(get_brain_area_pairs_group_delay, path, args=(
                multitaper_parameter_name, covariate, level,
                frequency_band, tetrode_pair_info))


def save_analysis(tetrode_info, tetrode_pair_info, area_pair_info,
                  ripple_info, label=''):
    RIPPLE_LOCKED_LEVELS = ['baseline', 'ripple_locked',
                            'ripple_difference_from_baseline']
    for multitaper_parameter_name in MULTITAPER_PARAMETERS:
        for level in RIPPLE_LOCKED_LEVELS:
            save_mean(label, multitaper_parameter_name, 'all_ripples',
                      level, tetrode_info, tetrode_pair_info,
                      area_pair_info)
    for multitaper_parameter_name in MULTITAPER_PARAMETERS:
        for covariate in RIPPLE_COVARIATES:
            for level in ripple_info[covariate].unique().tolist():
                save_mean(label, multitaper_parameter_name, covariate,
                          level, tetrode_info, tetrode_pair_info,
                          area_pair_info)
            for levels in combinations(
                    sorted(ripple_info[covariate].unique().tolist()), 2):
                level = '_'.join(levels)
                save_mean(label, multitaper_parameter_name, covariate,
                          level, tetrode_info, tetrode_pair_info,
                          area_pair_info)

if __name__ == '__main__':
    exit(main())
