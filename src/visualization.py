import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, gaussian

from loren_frank_data_processing import (get_multiunit_indicator_dataframe,
                                         get_spike_indicator_dataframe,
                                         reshape_to_segments)


def plot_perievent_raster(neuron_or_tetrode_key, animals, events, tetrode_info,
                          window_offset=(-0.5, 0.5),
                          sampling_frequency=1500, ax=None,
                          **scatter_kwargs):
    '''Plot spike raster relative to an event.

    Parameters
    ----------
    neuron_or_tetrode_key : tuple
    animals : dict of namedtuples
    events : pandas DataFrame, shape (n_events, 2)
    tetrode_info : pandas DataFrame, shape (n_tetrodes, ...)
    window_offset : tuple, optional
    sampling_frequency : tuple, optional
    ax : matplotlib axes, optional
    scatter_kwargs : dict

    Returns
    -------
    ax : matplotlib axes

    '''
    if ax is None:
        ax = plt.gca()
    try:
        spikes = get_spike_indicator_dataframe(neuron_or_tetrode_key, animals)
    except ValueError:
        spikes = ((get_multiunit_indicator_dataframe(
            neuron_or_tetrode_key, animals) > 0).sum(axis=1) > 0) * 1.0
    event_locked_spikes = reshape_to_segments(
        spikes, events, window_offset=window_offset,
        sampling_frequency=sampling_frequency).unstack(level=0).fillna(0)
    time = event_locked_spikes.index.total_seconds()
    spike_index, event_number = np.nonzero(event_locked_spikes.values)

    ax.scatter(time[spike_index], event_number, **scatter_kwargs)
    ax.axvline(0.0, color='black')
    ax.set_title(
        tetrode_info.loc[neuron_or_tetrode_key[:4]].area.upper() + ' - ' +
        str(neuron_or_tetrode_key))
    ax.set_ylabel(events.index.name)
    ax.set_xlabel('time (seconds)')
    ax.set_ylim((0, events.index.max() + 1))
    ax.set_xlim(window_offset)

    ax2 = ax.twinx()
    kde = kernel_density_estimate(
        event_locked_spikes, sampling_frequency, sigma=0.025)
    m = ax2.plot(time, kde[:, 1], color='blue', alpha=0.8)
    ax2.set_ylim((0, np.max([kde[:, 2].max(), 5])))
    ax2.fill_between(time, kde[:, 0], kde[:, 2],
                     color=m[0].get_color(), alpha=0.2)
    ax2.set_ylabel('Firing Rate (spikes / s)')

    return ax, ax2


def kernel_density_estimate(
        is_spike, sampling_frequency, sigma=0.025):
    '''The gaussian-smoothed kernel density estimate of firing rate over
    trials.

    Parameters
    ----------
    is_spike : ndarray, shape (n_time, n_trials)
    sampling_frequency : float
    bandwidth : float

    Returns
    -------
    firing_rate : ndarray, shape (n_time,)

    '''
    bandwidth = sigma * sampling_frequency
    n_window_samples = int(bandwidth * 8)
    kernel = gaussian(n_window_samples, bandwidth)[:, np.newaxis]
    density_estimate = convolve(
        is_spike, kernel, mode='same') / kernel.sum()
    n_events = density_estimate.shape[1]
    firing_rate = np.nanmean(density_estimate, axis=1,
                             keepdims=True) * sampling_frequency
    firing_rate_std = (np.nanstd(density_estimate, axis=1, keepdims=True) *
                       sampling_frequency / np.sqrt(n_events))
    ci = np.array([-1.96, 0, 1.96])
    return firing_rate + firing_rate_std * ci
