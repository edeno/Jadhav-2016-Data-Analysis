from functools import combinations, partial
from inspect import signature

import numpy as np
from scipy.ndimage import label
from scipy.stats import norm
from scipy.stats.mstats import linregress

from .minimum_phase_decomposition import minimum_phase_decomposition

EXPECTATION = {
    'trials': partial(np.mean, axis=1),
    'tapers': partial(np.mean, axis=2),
    'trials_tapers': partial(np.mean, axis=(1, 2))
}


class lazyproperty:

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class Connectivity(object):

    def __init__(self, fourier_coefficients, frequencies_of_interest=None,
                 expectation_type='trials_tapers'):
        self.fourier_coefficients = fourier_coefficients
        self.frequencies_of_interest = frequencies_of_interest
        self.expectation_type = expectation_type

    @lazyproperty
    def cross_spectral_matrix(self):
        '''

        Parameters
        ----------
        fourier_coefficients : array, shape (n_time_samples, n_trials,
                                             n_signals, n_fft_samples,
                                             n_tapers)

        Returns
        -------
        cross_spectral_matrix : array, shape (..., n_signals, n_signals)

        '''
        fourier_coefficients = (self.fourier_coefficients
                                .swapaxes(2, -1)[..., np.newaxis])
        return _complex_inner_product(fourier_coefficients,
                                      fourier_coefficients)

    @lazyproperty
    def power(self):
        fourier_coefficients = self.fourier_coefficients.swapaxes(2, -1)
        return self.expectation(fourier_coefficients *
                                fourier_coefficients.conjugate())

    @lazyproperty
    def minimum_phase_factor(self):
        return minimum_phase_decomposition(
            self.expectation(self.cross_spectral_matrix))

    @lazyproperty
    def transfer_function(self):
        return _estimate_transfer_function(self.minimum_phase_factor)

    @lazyproperty
    def noise_covariance(self):
        return _estimate_noise_covariance(self.minimum_phase_factor)

    @lazyproperty
    def MVAR_Fourier_coefficients(self):
        return np.linalg.inv(self.transfer_function)

    @property
    def expectation(self):
        return EXPECTATION[self.expectation_type]

    @property
    def n_observations(self):
        axes = signature(self.expectation).parameters['axis'].default
        if isinstance(axes, int):
            return self.cross_spectral_matrix.shape[axes]
        else:
            return np.prod(
                [self.cross_spectral_matrix.shape[axis]
                 for axis in axes])

    @property
    def bias(self):
        degrees_of_freedom = 2 * self.n_observations
        return 1 / (degrees_of_freedom - 2)

    def coherency(self):
        return self.expectation(self.cross_spectral_matrix) / np.sqrt(
            self.power[..., :, np.newaxis] *
            self.power[..., np.newaxis, :])

    def coherence_phase(self):
        return np.angle(self.coherency())

    def coherence_magnitude(self):
        return np.abs(self.coherency)

    def coherence_z(self):
        return fisher_z_transform(self.coherency(), self.bias)

    def coherence_p_values(self):
        return _get_normal_distribution_p_values(self.coherence_z())

    def imaginary_coherence(self):
        return np.abs(
            self.expectation(self.cross_spectral_matrix).imag /
            np.sqrt(self.power[..., :, np.newaxis] *
                    self.power[..., np.newaxis, :]))

    def canonical_coherence(self, group_labels):
        labels = np.unique(group_labels)
        normalized_fourier_coefficients = [
            _normalize_fourier_coefficients(
                self.fourier_coefficients[
                    :, :, np.in1d(group_labels, label), ...])
            for label in labels]
        return np.stack([
            _estimate_canonical_coherency(
                fourier_coefficients1, fourier_coefficients2)
            for fourier_coefficients1, fourier_coefficients2
            in combinations(normalized_fourier_coefficients, 2)
        ], axis=-1), list(combinations(labels, 2))

    def phase_locking_value(self):
        return self.expectation(
            self.cross_spectral_matrix /
            np.abs(self.cross_spectral_matrix))

    def phase_lag_index(self):
        return self.expectation(
            np.sign(self.cross_spectral_matrix.imag))

    def weighted_phase_lag_index(self):
        pli = self.phase_lag_index()
        weights = self.expectation(
            np.abs(self.cross_spectral_matrix.imag))
        with np.errstate(divide='ignore', invalid='ignore'):
            return pli / weights

    def debiased_squared_phase_lag_index(self):
        n_observations = self.n_observations
        return ((n_observations * self.phase_lag_index() ** 2 - 1.0) /
                (n_observations - 1.0))

    def debiased_squared_weighted_phase_lag_index(self):
        n_observations = self.n_observations
        imaginary_cross_spectral_matrix_sum = self.expectation(
            self.cross_spectral_matrix.imag) * n_observations
        squared_imaginary_cross_spectral_matrix_sum = self.expectation(
            self.cross_spectral_matrix.imag ** 2) * n_observations
        imaginary_cross_spectral_matrix_magnitude_sum = self.expectation(
            np.abs(self.cross_spectral_matrix.imag)) * n_observations
        weights = (imaginary_cross_spectral_matrix_magnitude_sum ** 2 -
                   squared_imaginary_cross_spectral_matrix_sum)
        return (imaginary_cross_spectral_matrix_sum ** 2 -
                squared_imaginary_cross_spectral_matrix_sum) / weights

    def pairwise_phase_consistency(self):
        n_observations = self.n_observations
        plv_sum = self.phase_locking_value() * n_observations
        ppc = ((plv_sum * plv_sum.conjugate() - n_observations) /
               (n_observations * (n_observations - 1.0)))
        return ppc.real

    def spectral_granger_prediction(self):
        partial_covariance = _remove_instantaneous_causality(
            self.noise_covariance)
        intrinsic_power = (self.power[..., np.newaxis] -
                           partial_covariance *
                           _magnitude(self.transfer_function))
        return _set_diagonal_to_zero(np.log(
            self.power[..., np.newaxis] / intrinsic_power))

    def directed_transfer_function(self, is_directed_coherence=False):
        if is_directed_coherence:
            noise_variance = np.diagonal(
                self.noise_covariance, axis1=-1, axis2=-2)[
                ..., np.newaxis, :, np.newaxis]
        else:
            noise_variance = 1.0

        transfer_magnitude = _magnitude(self.transfer_function)
        return (np.sqrt(noise_variance) * transfer_magnitude /
                _total_inflow(transfer_magnitude, noise_variance))

    def partial_directed_coherence(self, is_generalized=False):
        if is_generalized:
            noise_variance = np.diagonal(
                self.noise_covariance, axis1=-1, axis2=-2)[
                ..., np.newaxis, :, np.newaxis]
        else:
            noise_variance = 1.0
        return (self.MVAR_Fourier_coefficients *
                (1.0 / np.sqrt(noise_variance)) /
                _total_outflow(
                    self.MVAR_Fourier_coefficients, noise_variance))

    def direct_directed_transfer_function(self):
        transfer_magnitude = _magnitude(self.transfer_function)
        full_frequency_DTF = transfer_magnitude / np.sum(
            _total_inflow(transfer_magnitude, 1.0), axis=-3, keepdims=True)
        return full_frequency_DTF * self.partial_directed_coherence()

    def group_delay(self, frequencies_of_interest=None,
                    frequencies=None, frequency_resolution=None):
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution)
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest)
        is_significant = find_largest_significant_group(
            bandpassed_coherency, self.bias, independent_frequency_step)
        coherence_phase = np.ma.masked_array(
            np.unwrap(np.angle(bandpassed_coherency), axis=-3),
            mask=~is_significant)

        def _linear_regression(response):
            return linregress(bandpassed_frequencies, y=response)

        regression_results = np.ma.apply_along_axis(
            _linear_regression, -3, coherence_phase)
        slope = np.array(regression_results[..., 0, :, :], dtype=np.float)
        delay = slope / (2 * np.pi)
        r_value = np.array(
            regression_results[..., 2, :, :], dtype=np.float)
        return delay, slope, r_value

    def phase_slope_index(self, frequencies_of_interest=None,
                          frequencies=None, frequency_resolution=None):
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest)

        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution)
        frequency_index = np.arange(0, bandpassed_frequencies.shape[0],
                                    independent_frequency_step)
        bandpassed_coherency = bandpassed_coherency[
            ..., frequency_index, :, :]

        return np.imag(_inner_combination(bandpassed_coherency, axis=-1))


def _inner_combination(data, axis=-1):
    '''Takes the inner product of all possible pairs of a
    dimension without regard to order (combinations)'''
    combination_index = np.array(
        list(combinations(range(data.shape[axis]), 2)))
    return (np.take(data, combination_index[:, 0], axis).conjugate() *
            np.take(data, combination_index[:, 1], axis)).sum(axis=axis)


def _estimate_noise_covariance(minimum_phase):
    A_0 = minimum_phase[..., 0, :, :]
    return np.matmul(A_0, A_0.swapaxes(-1, -2)).real


def _estimate_transfer_function(minimum_phase):
    return np.matmul(minimum_phase,
                     np.linalg.inv(minimum_phase[..., 0:1, :, :]))


def _magnitude(x):
    return np.abs(x) ** 2


def _complex_inner_product(a, b):
    return np.matmul(a, _conjugate_transpose(b))


def _remove_instantaneous_causality(noise_covariance):
    noise_covariance = noise_covariance[..., np.newaxis, :, :]
    variance = np.diagonal(noise_covariance, axis1=-1,
                           axis2=-2)[..., np.newaxis]
    return (_conjugate_transpose(variance) -
            noise_covariance * _conjugate_transpose(noise_covariance) /
            variance)


def _set_diagonal_to_zero(x):
    n_signals = x.shape[-1]
    diagonal_index = np.diag_indices(n_signals)
    x[..., diagonal_index[0], diagonal_index[1]] = 0
    return x


def _total_inflow(transfer_magnitude, noise_variance):
    return np.sum(noise_variance * transfer_magnitude,
                  keepdims=True, axis=-1)


def _get_noise_variance(noise_covariance):
    if noise_covariance is None:
        return 1.0
    else:
        return np.diagonal(noise_covariance, axis1=-1, axis2=-2)[
            ..., np.newaxis]


def _total_outflow(MVAR_Fourier_coefficients, noise_variance):
    return np.sum(
        (1.0 / noise_variance) * _magnitude(MVAR_Fourier_coefficients),
        keepdims=True, axis=-2)


def _reshape(fourier_coefficients):
    '''Combine trials and tapers dimensions'''
    (n_time_samples, _, n_signals,
     n_fft_samples, _) = fourier_coefficients.shape
    return fourier_coefficients.swapaxes(1, 3).reshape(
        (n_time_samples, n_fft_samples, n_signals, -1))


def _normalize_fourier_coefficients(fourier_coefficients):
    U, _, V = np.linalg.svd(
        _reshape(fourier_coefficients), full_matrices=False)
    return np.matmul(U, V)


def _estimate_canonical_coherency(normalized_fourier_coefficients1,
                                  normalized_fourier_coefficients2):
    group_cross_spectrum = _complex_inner_product(
        normalized_fourier_coefficients1, normalized_fourier_coefficients2)
    return np.linalg.svd(group_cross_spectrum,
                         full_matrices=False, compute_uv=False)[..., 0]


def _bandpass(data, frequencies, frequencies_of_interest, axis=-3):
    frequency_index = ((frequencies_of_interest[0] < frequencies) &
                       (frequencies < frequencies_of_interest[1]))
    return (np.take(data, frequency_index, axis=axis),
            frequencies[frequency_index])


def _get_independent_frequency_step(frequency_difference,
                                    frequency_resolution):
    '''Find the number of points of a frequency axis such that they
    are statistically independent.


    Parameters
    ----------
    frequency_difference : float
        The distance between two frequency points
    frequency_resolution : float
        The ability to resolve frequency points

    Returns
    -------
    frequency_step : int
        The number of points required so that two
        frequency points are statistically independent.
    '''
    return np.ceil(frequency_resolution / frequency_difference).astype(int)




def _find_largest_group(is_significant):
    labeled, _ = label(is_significant)
    label_groups, label_counts = np.unique(labeled, return_counts=True)

    if len(label_groups) > 1:
        label_counts[0] = 0
        max_group = label_groups[np.argmax(label_counts)]
        return labeled == max_group
    else:
        return np.zeros(is_significant.shape, dtype=bool)


def _filter_by_frequency_resolution(is_significant, frequency_step):
    index = is_significant.nonzero()[0]
    independent_index = index[slice(0, len(index), frequency_step)]
    return np.in1d(np.arange(0, len(is_significant)), independent_index)


def _find_largest_independent_group(is_significant, frequency_step,
                                    smallest_group_size=3):
    is_significant = _filter_by_frequency_resolution(
        _find_largest_group(is_significant),
        frequency_step)
    if sum(is_significant) >= smallest_group_size:
        return is_significant
    else:
        return np.zeros(is_significant.shape, dtype=bool)


def find_largest_significant_group(coherency, bias, frequency_step=1,
                                   significance_threshold=0.05,
                                   smallest_group_size=3):
    z_coherence = fisher_z_transform(coherency, bias)
    p_values = get_normal_distribution_p_values(z_coherence)
    is_significant = adjust_for_multiple_comparisons(
        p_values, alpha=significance_threshold)
    return np.apply_along_axis(_find_largest_independent_group, -3,
                               is_significant, frequency_step,
                               smallest_group_size)


def _conjugate_transpose(x):
    '''Conjugate transpose of the last two dimensions of array x'''
    return x.swapaxes(-1, -2).conjugate()
