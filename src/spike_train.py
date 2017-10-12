import numpy as np
import pandas as pd
import xarray as xr
from patsy import build_design_matrices, dmatrix
from scipy.signal import convolve, gaussian, correlate
from scipy.stats import poisson
from statsmodels.api import GLM, families


def perievent_time_kernel_density_estimate(
        is_spike, sampling_frequency, bandwidth=30):
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
    kernel = gaussian(bandwidth * 5, bandwidth)[:, np.newaxis]
    density_estimate = convolve(
        is_spike, kernel, mode='same') / kernel.sum()
    return np.nanmean(density_estimate, axis=1) * sampling_frequency


def perievent_time_spline_estimate(is_spike, time, sampling_frequency,
                                   formula='bs(time, df=5)',
                                   n_boot_samples=None, trial_id=None):
    design_matrix = dmatrix(formula, dict(time=time),
                            return_type='dataframe')
    fit = GLM(is_spike, design_matrix, family=families.Poisson()).fit()

    if n_boot_samples is not None:
        model_coefficients = glm_parametric_bootstrap(
            fit.params, fit.cov_params(),
            n_samples=n_boot_samples)
    else:
        model_coefficients = fit.params[:, np.newaxis]
        n_boot_samples = 1

    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], dict(time=np.unique(time)))[0]

    coords = {'time': np.unique(time)}

    firing_rate = xr.DataArray(
        np.exp(np.dot(predict_design_matrix, model_coefficients)) *
        sampling_frequency, dims=['time', 'n_boot_samples'],
        coords=coords, name='firing_rate')
    multiplicative_gain = xr.DataArray(np.exp(
        np.dot(predict_design_matrix[:, 1:], model_coefficients[1:])),
        dims=['time', 'n_boot_samples'],
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(np.exp(
        model_coefficients[0]) * sampling_frequency,
        dims='n_boot_samples', name='baseline_firing_rate')

    conditional_intensity = np.exp(
        np.dot(design_matrix, model_coefficients))
    ks_statistic = xr.DataArray(
        [TimeRescaling(ci, is_spike, trial_id,
                       adjust_for_short_trials=True).ks_statistic()
         for ci in conditional_intensity.T],
        dims='n_boot_samples', name='ks_statistic')

    return xr.merge((firing_rate, multiplicative_gain,
                     baseline_firing_rate, ks_statistic))


def perievent_time_indicator_estimate(is_spike, time, sampling_frequency,
                                      formula='time', n_boot_samples=1000,
                                      trial_id=None):
    time_indicator = np.where(time > 0, 'after0', 'before0')
    time_indicator = pd.Categorical(
        time_indicator, categories=['before0', 'after0'], ordered=True)
    design_matrix = dmatrix(
        'time', {'time': time_indicator}, return_type='dataframe')

    fit = GLM(is_spike, design_matrix, family=families.Poisson()).fit()

    if n_boot_samples is not None:
        model_coefficients = glm_parametric_bootstrap(
            fit.params, fit.cov_params(),
            n_samples=n_boot_samples)
    else:
        model_coefficients = fit.params[:, np.newaxis]
        n_boot_samples = 1

    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info],
        dict(time=np.unique(time_indicator)))[0]

    firing_rate_change = -1 * np.diff(
        np.exp(np.dot(predict_design_matrix, model_coefficients)) *
        sampling_frequency, axis=0).squeeze()
    multiplicative_change = np.exp(model_coefficients[1]).squeeze()
    before0_firing_rate = np.exp(
        model_coefficients[0]) * sampling_frequency

    data_vars = {
        'multiplicative_change_after0_vs_before0': (
            ['n_boot_samples'], multiplicative_change),
        'firing_rate_change_after0_vs_before0': (
            ['n_boot_samples'], firing_rate_change),
        'before0_firing_rate': (['n_boot_samples'], before0_firing_rate),
    }
    coords = {'n_boot_samples': np.arange(n_boot_samples) + 1}
    return xr.Dataset(data_vars, coords)


def glm_parametric_bootstrap(model_coefficients, model_covariance_matrix,
                             n_samples=1000):
    return np.random.multivariate_normal(
        model_coefficients, model_covariance_matrix, n_samples).T


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


def coherence_rate_adjustment(firing_rate_condition1,
                              firing_rate_condition2, spike_autospectrum,
                              homogeneous_poisson_noise=0, dt=1):
    '''Correction for the spike-field or spike-spike coherence when the
    conditions have different firing rates.

    When comparing the coherence of two conditions, a change in firing rate
    results in a change in coherence without an increase in coupling.
    This adjustment modifies the coherence of one of the conditions, so
    that a difference in coherence between conditions indicates a change
    in coupling, not firing rate. See [1] for details.

    If using to compare spike-spike coherence, not that the coherence
    adjustment must be applied twice, once for each spike train.

    Adjusts `firing_rate_condition1` to `firing_rate_condition2`.

    Parameters
    ----------
    firing_rate_condition1, firing_rate_condition2 : float
        Average firing rates for each condition.
    spike_autospectrum : ndarray, shape (n_frequencies,)
        Autospectrum of the spike train. Complex.
    homogeneous_poisson_noise : float, optional
        Beta in [1].
    dt : float, optional
        Size of time step.

    Returns
    -------
    rate_adjustment_factor : ndarray, shape (n_frequencies)

    References
    ----------
    .. [1] Aoi, M.C., Lepage, K.Q., Kramer, M.A., and Eden, U.T. (2015).
           Rate-adjusted spike-LFP coherence comparisons from spike-train
           statistics. Journal of Neuroscience Methods 240, 141-153.

    '''
    # alpha in [1]
    firing_rate_ratio = firing_rate_condition2 / firing_rate_condition1
    adjusted_firing_rate = (
        (1 / firing_rate_ratio - 1) * firing_rate_condition1 +
        homogeneous_poisson_noise / firing_rate_ratio ** 2) * dt ** 2
    return 1 / np.sqrt(1 + (adjusted_firing_rate / spike_autospectrum))
