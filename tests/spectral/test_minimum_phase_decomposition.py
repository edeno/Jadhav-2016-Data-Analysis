import numpy as np
from scipy.signal import freqz_zpk

from src.spectral.minimum_phase_decomposition import (
    _check_convergence, _conjugate_transpose, _get_causal_signal,
    minimum_phase_decomposition)


def test__check_convergence():
    tolerance = 1e-8
    n_time_points = 5
    minimum_phase_factor = np.zeros((n_time_points, 4, 3))
    old_minimum_phase_factor = np.zeros((n_time_points, 4, 3))
    minimum_phase_factor[0, :, :] = 1e-9
    minimum_phase_factor[1, :, :] = 1e-7
    minimum_phase_factor[3, :] = 1
    minimum_phase_factor[4, :3, 1:2] = 1e-7

    expected_is_converged = np.array([True, False, True, False, False])

    is_converged = _check_convergence(
        minimum_phase_factor, old_minimum_phase_factor, tolerance)

    assert np.all(is_converged == expected_is_converged)


def test__conjugate_transpose():
    test_array = np.zeros((2, 2, 4), dtype=np.complex)
    test_array[1, ...] = [[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
                          [1 - 2j, 3 - 4j, 5 - 6j, 7 - 8j]]
    expected_array = np.zeros((2, 4, 2), dtype=np.complex)
    expected_array[1, ...] = test_array[1, ...].conj().transpose()
    assert np.allclose(_conjugate_transpose(test_array), expected_array)


def test_minimum_phase_decomposition():
    n_signals = 1
    # minimum phase is all poles and zeros inside the unit circle
    zero, pole, gain = 0.25, 0.50, 2.00
    _, transfer_function = freqz_zpk(zero, pole, gain, whole=True)
    n_fft_samples = transfer_function.shape[0]
    expected_minimum_phase_factor = np.zeros(
        (1, n_fft_samples, n_signals, n_signals), dtype=np.complex)
    expected_minimum_phase_factor[
        0, :n_fft_samples, 0, 0] = transfer_function

    cross_spectral_matrix = np.matmul(
        expected_minimum_phase_factor,
        _conjugate_transpose(expected_minimum_phase_factor))
    minimum_phase_factor = minimum_phase_decomposition(
        cross_spectral_matrix)

    assert np.all(_check_convergence(
        minimum_phase_factor, expected_minimum_phase_factor))
    assert np.all(_check_convergence(
        minimum_phase_factor *
        _conjugate_transpose(minimum_phase_factor), cross_spectral_matrix))
