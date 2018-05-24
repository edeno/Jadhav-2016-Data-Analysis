'''Replicate analysis in Jadhav 2016.
'''
from logging import getLogger

import numpy as np
import pandas as pd

from loren_frank_data_processing import (get_spike_indicator_dataframe,
                                         make_neuron_dataframe,
                                         reshape_to_segments)
from src.analysis import detect_epoch_ripples

logger = getLogger(__name__)


def _get_trial_spikes(neuron_key, animals, events,
                      window_offset=(-0.5, 0.5),
                      sampling_frequency=1500):
    spikes = get_spike_indicator_dataframe(neuron_key, animals)
    return reshape_to_segments(
        spikes, events, window_offset=window_offset,
        sampling_frequency=sampling_frequency).unstack(level=0).fillna(0)


def _jitter(ripple_locked_spikes):
    n_time = ripple_locked_spikes.shape[0]
    return np.stack(
        [np.roll(trial_spikes, np.random.randint(-n_time, n_time))
         for trial_spikes in ripple_locked_spikes.T], axis=1)


def _mean_squared_diff(x, y, axis=None):
    return np.mean((x - y) ** 2, axis=axis)


def sharp_wave_ripple_modulation(neuron_key, animals, ripple_times,
                                 sampling_frequency, n_shuffle=5000):
    logger.info(neuron_key)
    ripple_locked_spikes = _get_trial_spikes(neuron_key, animals, ripple_times)
    time = ripple_locked_spikes.index.total_seconds()
    is_after_swr = (time >= 0.000) & (time <= 0.200)
    is_pre_swr = (time >= -0.500) & (time <= -0.100)
    psth = ripple_locked_spikes.values.mean(axis=1)
    modulation = (np.mean(psth[is_after_swr])
                  - np.mean(psth[is_pre_swr])) * sampling_frequency
    zscore = (psth - np.mean(psth)) / np.std(psth)
    rise_fall_time = time[np.nonzero(np.abs(zscore) > 1)[0][0]]
    peak_time = time[np.argmax(psth)]
    trough_time = time[np.argmin(psth)]

    if ripple_locked_spikes.values.sum() > 50:
        shuffled_psth = np.stack(
            [_jitter(ripple_locked_spikes.values).mean(axis=1)
             for _ in range(n_shuffle)], axis=1)[is_after_swr]
        baseline = shuffled_psth.mean(axis=1)
        null = _mean_squared_diff(shuffled_psth, baseline[:, None], axis=1)
        test_stat = _mean_squared_diff(psth[is_after_swr], baseline)
        min_null, max_null = np.percentile(null, [2.5, 97.5])
        is_significant = (test_stat < min_null) | (test_stat > max_null)
        return (is_significant.astype(np.float), test_stat,
                modulation, rise_fall_time, peak_time, trough_time)
    else:
        return (np.nan, np.nan, modulation, rise_fall_time,
                peak_time, trough_time)


def swr_stats(epoch_key, animals, sampling_frequency):
    ripple_times = detect_epoch_ripples(
        epoch_key, animals, sampling_frequency)
    neuron_info = make_neuron_dataframe(animals).xs(
        epoch_key, drop_level=False)
    is_PFC_FS_interneuron = (
        (neuron_info.meanrate > 17.0) & (neuron_info.spikewidth < 0.3))
    is_CA1_FS_interneuron = (
        (neuron_info.meanrate > 7.0) & (neuron_info.spikewidth < 0.3))
    neuron_info = neuron_info.loc[
        (neuron_info.area.isin(['PFC']) & ~is_PFC_FS_interneuron) |
        (neuron_info.area.isin(['CA1', 'iCA1']) & ~is_CA1_FS_interneuron) &
        (neuron_info.numspikes > 0)
    ]

    stats = np.array(
        [sharp_wave_ripple_modulation(neuron_key, animals, ripple_times,
                                      sampling_frequency)
         for neuron_key in neuron_info.index])
    COLUMNS = ['is_significant', 'diff_from_baseline', 'swr_modulation',
               'swr_onset', 'peak_time', 'trough_time']

    stats = pd.DataFrame(stats, index=neuron_info.index,
                         columns=COLUMNS)

    return pd.concat((neuron_info, stats), axis=1)
