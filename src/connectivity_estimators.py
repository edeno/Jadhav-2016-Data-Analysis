from functools import partial
from inspect import signature
from itertools import combinations

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import detrend

EXPECTATION = {
    'trials': partial(np.mean, axis=1),
    'tapers': partial(np.mean, axis=2),
    'trials_tapers': partial(np.mean, axis=(1, 2))
}


def _conjugate_transpose(x):
    '''Conjugate transpose of the last two dimensions of array x'''
    return x.swapaxes(-1, -2).conjugate()


def _complex_inner_product(a, b):
    return np.matmul(a, _conjugate_transpose(b))


def compute_cross_spectral_matrix(fourier_coefficients):
    '''
    Parameters
    ----------
    fourier_coefficients : array-like, shape (n_time_samples, n_trials,
                                              n_signals, n_fft_samples,
                                              n_tapers)

    Returns
    -------
    cross_spectral_matrix : array-like, shape (n_time_samples, n_trials,
                                               n_tapers, n_fft_samples,
                                               n_signals, n_signals)
    '''
    fourier_coefficients = np.expand_dims(
        fourier_coefficients.swapaxes(2, -1), -1)
    return _complex_inner_product(
        fourier_coefficients, fourier_coefficients)


def power(cross_spectral_matrix=None, fourier_coefficients=None,
          expectation=EXPECTATION['trials_tapers']):
    if cross_spectral_matrix is None:
        fourier_coefficients = fourier_coefficients.swapaxes(2, -1)
        auto_spectra = expectation(fourier_coefficients *
                                   fourier_coefficients.conjugate())
    else:
        auto_spectra = expectation(
            np.diagonal(cross_spectral_matrix, 0, -2, -1))
    return auto_spectra.real


def coherency(cross_spectral_matrix,
              expectation=EXPECTATION['trials_tapers']):
    power_spectra = power(cross_spectral_matrix=cross_spectral_matrix,
                          expectation=expectation)
    return expectation(cross_spectral_matrix) / np.sqrt(
        power_spectra[..., :, np.newaxis] *
        power_spectra[..., np.newaxis, :])


def imaginary_coherence(cross_spectral_matrix,
                        expectation=EXPECTATION['trials_tapers']):
    auto_spectra = power(cross_spectral_matrix=cross_spectral_matrix,
                         expectation=expectation)
    return np.abs(expectation(cross_spectral_matrix).imag /
                  np.sqrt(auto_spectra[:, :, :, np.newaxis] *
                          auto_spectra[:, :, np.newaxis, :]))


def phase_locking_value(cross_spectral_matrix,
                        expectation=EXPECTATION['trials_tapers']):
    return expectation(cross_spectral_matrix /
                       np.abs(cross_spectral_matrix))


def phase_lag_index(cross_spectral_matrix,
                    expectation=EXPECTATION['trials_tapers']):
    return expectation(np.sign(cross_spectral_matrix.imag))


def weighted_phase_lag_index(cross_spectral_matrix,
                             expectation=EXPECTATION['trials_tapers']):
    pli = phase_lag_index(cross_spectral_matrix, expectation=expectation)
    weights = expectation(np.abs(cross_spectral_matrix.imag))
    with np.errstate(divide='ignore', invalid='ignore'):
        return pli / weights


def _get_number_observations(cross_spectral_matrix, expectation_function):
    return np.prod(
        [cross_spectral_matrix.shape[axis] for axis
         in signature(expectation_function).parameters['axis'].default])


def debiased_squared_phase_lag_index(
        cross_spectral_matrix, expectation=EXPECTATION['trials_tapers']):
    n_observations = _get_number_observations(
        cross_spectral_matrix, expectation)
    pli = phase_lag_index(cross_spectral_matrix, expectation=expectation)
    return (n_observations * pli ** 2 - 1.0) / (n_observations - 1.0)


def debiased_squared_weighted_phase_lag_index(
        cross_spectral_matrix, expectation=EXPECTATION['trials_tapers']):
    n_observations = _get_number_observations(cross_spectral_matrix,
                                              expectation)
    imaginary_cross_spectral_matrix_sum = expectation(
        cross_spectral_matrix.imag) * n_observations
    squared_imaginary_cross_spectral_matrix_sum = expectation(
        cross_spectral_matrix.imag ** 2) * n_observations
    imaginary_cross_spectral_matrix_magnitude_sum = expectation(
        np.abs(cross_spectral_matrix.imag)) * n_observations
    weights = (imaginary_cross_spectral_matrix_magnitude_sum ** 2 -
               squared_imaginary_cross_spectral_matrix_sum)
    return (imaginary_cross_spectral_matrix_sum ** 2 -
            squared_imaginary_cross_spectral_matrix_sum) / weights


def pairwise_phase_consistency(
        cross_spectral_matrix, expectation=EXPECTATION['trials_tapers']):
    n_observations = _get_number_observations(
        cross_spectral_matrix, expectation)
    plv_sum = phase_locking_value(
        cross_spectral_matrix, expectation=expectation) * n_observations
    ppc = ((plv_sum * plv_sum.conjugate() - n_observations) /
           (n_observations * (n_observations - 1.0)))
    return ppc.real


def _get_intial_conditions(cross_spectral_matrix):
    '''Returns a guess for the minimum phase factor'''
    return np.linalg.cholesky(
        ifft(cross_spectral_matrix, axis=-3)[..., 0:1, :, :].real
    ).swapaxes(-1, -2)


def _get_causal_signal(linear_predictor):
    '''Remove negative lags (the anti-casual part of the signal) and half
    of the zero lag to obtain the causal signal.
    Gives you A_(t+1)(Z) / A_(t)(Z)

    Takes half the roots on the unit circle (zero lag) and all the roots
    inside the unit circle (positive lags)

    This is the plus operator in Wilson
    '''
    n_signals, n_fft_samples = (linear_predictor.shape[-1],
                                linear_predictor.shape[-3])
    linear_predictor_coefficients = ifft(linear_predictor, axis=-3)
    linear_predictor_coefficients[..., 0, :, :] *= 0.5
    # Form S_tau
    lower_triangular_ind = np.tril_indices(n_signals, k=-1)
    linear_predictor_coefficients[
        ..., 0, lower_triangular_ind[0], lower_triangular_ind[1]] = 0
    linear_predictor_coefficients[..., (n_fft_samples // 2) + 1:, :, :] = 0
    return fft(linear_predictor_coefficients, axis=-3)


def _check_convergence(minimum_phase_factor, old_minimum_phase_factor,
                       tolerance):
    '''Check convergence of Wilson algorithm at each time point'''
    n_time_points = minimum_phase_factor.shape[0]
    psi_error = np.linalg.norm(
        np.reshape(minimum_phase_factor -
                   old_minimum_phase_factor, (n_time_points, -1)),
        ord=np.inf, axis=1)
    return psi_error < tolerance


def minimum_phase_decomposition(cross_spectral_matrix, tolerance=1E-8,
                                max_iterations=30):
    '''Using the Wilson algorithm to find a minimum phase matrix square
    root of the cross spectral density'''
    n_time_points, n_signals = (cross_spectral_matrix.shape[0],
                                cross_spectral_matrix.shape[-1])
    I = np.eye(n_signals)
    is_converged = np.zeros(n_time_points, dtype=bool)
    minimum_phase_factor = np.zeros(cross_spectral_matrix.shape)
    minimum_phase_factor[..., :, :, :] = _get_intial_conditions(
        cross_spectral_matrix)

    for iteration in range(max_iterations):
        old_minimum_phase_factor = minimum_phase_factor.copy()
        linear_predictor = (np.linalg.solve(
            minimum_phase_factor,
            _conjugate_transpose(np.linalg.solve(minimum_phase_factor,
                                                 cross_spectral_matrix)))
                            + I)
        minimum_phase_factor = np.matmul(
            minimum_phase_factor, _get_causal_signal(linear_predictor))

        minimum_phase_factor[is_converged, ...] = old_minimum_phase_factor[
            is_converged, ...]
        is_converged = _check_convergence(
            minimum_phase_factor, old_minimum_phase_factor, tolerance)
        if np.all(is_converged):
            return minimum_phase_factor
    else:
        print('Maximum iterations reached. {} of {} converged'.format(
            is_converged.sum(), len(is_converged)))
        return minimum_phase_factor


def _estimate_noise_covariance(minimum_phase):
    A_0 = minimum_phase[..., 0, :, :]
    return np.matmul(A_0, A_0.swapaxes(-1, -2)).real


def _estimate_transfer_function(minimum_phase):
    return np.matmul(minimum_phase,
                     np.linalg.inv(minimum_phase[..., 0:1, :, :]))


def _transfer_magnitude(transfer_function):
    return np.abs(transfer_function) ** 2


def _remove_instantaneous_causality(noise_covariance):
    noise_covariance = noise_covariance[..., np.newaxis, :, :]
    variance = np.diagonal(noise_covariance, axis1=-1, axis2=-2)[
        ..., np.newaxis]
    return (_conjugate_transpose(variance) -
            noise_covariance * _conjugate_transpose(noise_covariance) /
            variance)


def _set_diagonal_to_zero(x):
    n_signals = x.shape[-1]
    diagonal_index = np.diag_indices(n_signals)
    x[..., diagonal_index[0], diagonal_index[1]] = 0
    return x


def spectral_granger_prediction(power_spectra, transfer_function,
                                noise_covariance):
    transfer_magnitude = _transfer_magnitude(transfer_function)
    partial_covariance = _remove_instantaneous_causality(noise_covariance)
    power_spectra = power_spectra[..., np.newaxis]
    intrinsic_power = (power_spectra -
                       partial_covariance * transfer_magnitude)
    return _set_diagonal_to_zero(np.log(power_spectra / intrinsic_power))


def _total_inflow(transfer_magnitude, noise_variance):
    return np.sqrt(np.sum(
        noise_variance * transfer_magnitude, keepdims=True, axis=-1))


def _get_noise_variance(noise_covariance):
    if noise_covariance is None:
        return 1.0
    else:
        return np.diagonal(noise_covariance, axis1=-1, axis2=-2)[
            ..., np.newaxis]


def directed_transfer_function(transfer_function, noise_covariance=None):
    transfer_magnitude = _transfer_magnitude(transfer_function)
    noise_variance = _get_noise_variance(noise_covariance)
    return np.sqrt(noise_variance) * transfer_magnitude / _total_inflow(
        transfer_magnitude, noise_variance)


def _total_outflow(MVAR_Fourier_coefficients, noise_variance):
    return np.sqrt(np.sum(
        (1.0 / noise_variance) * np.abs(MVAR_Fourier_coefficients) ** 2,
        keepdims=True, axis=-2))


def partial_directed_coherence(MVAR_Fourier_coefficients,
                               noise_covariance=None):
    noise_variance = _get_noise_variance(noise_covariance)
    return (MVAR_Fourier_coefficients * (1.0 / np.sqrt(noise_variance)) /
            _total_outflow(MVAR_Fourier_coefficients, noise_variance))


def direct_directed_transfer_function(transfer_function):
    transfer_magnitude = _transfer_magnitude(transfer_function)
    full_frequency_DTF = transfer_magnitude / np.sum(
        _total_inflow(transfer_magnitude, 1.0), axis=-3, keepdims=True)
    partial_coherence = partial_directed_coherence(
        transfer_function=transfer_function)
    return full_frequency_DTF * partial_coherence


def _reshape(fourier_coefficients):
    n_signals = len(fourier_coefficients)
    n_time_samples, _, n_fft_samples, _ = fourier_coefficients[0].shape
    return np.stack(fourier_coefficients, axis=2).swapaxes(1, 3).reshape(
        (n_time_samples, n_fft_samples, n_signals, -1))


def _normalize_fourier_coefficients(fourier_coefficients):
    U, _, V = np.linalg.svd(
        _reshape(fourier_coefficients), full_matrices=False)
    return np.matmul(U, V)


def _estimate_canonical_coherency(fourier_coefficients1,
                                  fourier_coefficients2):
    group_cross_spectrum = _complex_inner_product(
        fourier_coefficients1, fourier_coefficients2)
    return np.linalg.svd(
        group_cross_spectrum, full_matrices=False, compute_uv=False)[
            ..., 0]


def canonical_coherence(fourier_coefficients, group_labels):
    normalized_fourier_coefficients = [
        _normalize_fourier_coefficients(
            fourier_coefficients[:, :, group_labels == group_ID, ...])
        for group_ID in np.unique(group_labels)]
    return [_estimate_canonical_coherency(fourier_coefficients1,
                                          fourier_coefficients2)
            for fourier_coefficients1, fourier_coefficients2
            in combinations(normalized_fourier_coefficients, 2)]


def _linear_regression(frequencies, coherence_phase, axis=-3):
    frequencies = detrend(frequencies, type='constant')[
        ..., np.newaxis,   np.newaxis]
    coherence_phase = detrend(coherence_phase, axis=axis, type='constant')
    cov = np.sum(frequencies * coherence_phase, axis=axis)
    var_x = np.sum(frequencies ** 2)
    var_y = np.sum(coherence_phase ** 2, axis=axis)
    slope = cov / var_x
    correlation = cov ** 2 / (var_x * var_y)
    delay = slope / (2 * np.pi)
    return delay, slope, correlation


def group_delay(bandpassed_coherency, frequencies):
    coherence_phase = np.unwrap(np.angle(bandpassed_coherency), axis=-3)
    try:
        delay, slope, correlation = _linear_regression(
            frequencies, coherence_phase)
    except ValueError:
        delay, slope, correlation = 3 * (
            np.full(bandpassed_coherency.shape, np.nan),)
    return delay, correlation, slope


def phase_slope_index(bandpassed_coherency):
    phase_slope = (bandpassed_coherency[..., :, np.newaxis, :, :] *
                   bandpassed_coherency[..., np.newaxis, :, :, :].conj())
    n_signals = bandpassed_coherency.shape[-1]
    lower_triangular_indicies = np.tril_indices(n_signals)
    phase_slope[
        ..., lower_triangular_indicies[0],
        lower_triangular_indicies[1], :, :] = 0
    return phase_slope.sum(axis=(-3, -4)).imag
