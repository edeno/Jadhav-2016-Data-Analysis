import numpy as np
import pandas as pd
import xarray as xr
from patsy import build_design_matrices, dmatrix
from scipy.signal import convolve, gaussian, correlate
from statsmodels.api import GLM, families
from time_rescale import TimeRescaling


def kernel_density_estimate(
        is_spike, sampling_frequency, sigma=0.025):
    '''The gaussian-smoothed kernel density estimate of firing rate over
    trials.

    Parameters
    ----------
    is_spike : ndarray, shape (n_time, n_trials)
    sampling_frequency : float
    sigma : float

    Returns
    -------
    firing_rate : ndarray, shape (n_time,)

    '''
    bandwidth = sigma * sampling_frequency
    n_window_samples = int(bandwidth * 8)
    kernel = gaussian(n_window_samples, bandwidth)[:, np.newaxis]
    density_estimate = convolve(
        is_spike, kernel, mode='same') / kernel.sum()
    return np.nanmean(density_estimate, axis=1) * sampling_frequency


def cross_correlate(spike_train1, spike_train2=None, sampling_frequency=1):
    '''

    Parameters
    ----------
    spike_train1 : ndarray, shape (n_time, n_trials)
    spike_train2 : ndarray or None, shape (n_time, n_trials)
        If None, the autocorrelation of spike_train1 is computed.
    sampling_frequency : float, optional

    Returns
    -------
    cross_correlation : pandas Series

    '''
    if spike_train2 is None:
        spike_train2 = spike_train1.copy()
    correlation = np.array(
        [correlate(spike_train1_by_trial, spike_train2_by_trial)
         for spike_train1_by_trial, spike_train2_by_trial
         in zip(spike_train1.T, spike_train2.T)]).T
    correlation = np.nanmean(correlation / correlation.max(axis=0), axis=1)
    n_time = spike_train1.shape[0]
    dt = 1 / sampling_frequency
    delay = pd.Index(dt * np.arange(-n_time + 1, n_time), name='delay')
    return pd.Series(correlation, index=delay, name='correlation')


def simulate_poisson_process(rate, sampling_frequency):
    '''

    Parameters
    ----------
    rate : ndarray
    sampling_frequency : float

    Returns
    -------
    poisson_point_process : ndarray
        Same shape as rate.
    '''
    return np.random.poisson(rate / sampling_frequency)
