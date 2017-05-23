import numpy as np
from src.spectral.connectivity import Connectivity
from pytest import mark


@mark.parametrize('axis', [(0), (1), (2), (3)])
def test_cross_spectrum(axis):
    '''Test that the cross spectrum is correct for each dimension.'''
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        2, 2, 2, 2, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    signal_fourier_coefficient = [2 * np.exp(1j * np.pi / 2),
                                  3 * np.exp(1j * -np.pi / 2)]
    fourier_ind = [slice(0, 4)] * 5
    fourier_ind[-1] = slice(None)
    fourier_ind[axis] = slice(1, 2)
    fourier_coefficients[fourier_ind] = signal_fourier_coefficient

    expected_cross_spectral_matrix = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals,
         n_signals), dtype=np.complex)

    expected_slice = np.array([[4, -6], [-6, 9]], dtype=np.complex)
    expected_ind = [slice(0, 5)] * 6
    expected_ind[-1] = slice(None)
    expected_ind[-2] = slice(None)
    expected_ind[axis] = slice(1, 2)
    expected_cross_spectral_matrix[expected_ind] = expected_slice

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        expected_cross_spectral_matrix, this_Conn.cross_spectral_matrix)


def test_power():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 1, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * np.pi / 2),
                                    3 * np.exp(1j * -np.pi / 2)]

    expected_power = np.zeros((n_time_samples, n_fft_samples, n_signals))

    expected_power[..., :] = [4, 9]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        expected_power, this_Conn.power)


@mark.parametrize(
    'expectation_type, expected_shape',
    [('trials_tapers', (1, 4, 5)),
     ('trials', (1, 3, 4, 5)),
     ('tapers', (1, 2, 4, 5))])
def test_expectation(expectation_type, expected_shape):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    expectation_function = this_Conn.expectation
    assert np.allclose(
        expected_shape, expectation_function(fourier_coefficients).shape)


@mark.parametrize(
    'expectation_type, expected_n_observations',
    [('trials_tapers', 6),
     ('trials', 2),
     ('tapers', 3)])
def test_n_observations(expectation_type, expected_n_observations):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    assert this_Conn.n_observations == expected_n_observations


@mark.parametrize(
    'expectation_type, expected_bias',
    [('trials_tapers', 0.10),
     ('trials', 0.50),
     ('tapers', 0.25)])
def test_bias(expectation_type, expected_bias):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    assert np.allclose(this_Conn.bias, expected_bias)


def test_coherency():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * np.pi / 2),
                                    3 * np.exp(1j * -np.pi / 2)]
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    expected_coherence_magnitude = [[4, 1], [1, 9]]
    expected_phase = np.zeros((2, 2))
    expected_phase[0, 1] = np.pi
    expected_phase[1, 0] = -np.pi

    assert np.allclose(
        np.abs(this_Conn.coherency().squeeze()),
        expected_coherence_magnitude)
    assert np.allclose(
        np.angle(this_Conn.coherency().squeeze()),
        expected_phase)


def test_imaginary_coherence():
    '''Test that imaginary coherence sets signals with the same phase
    to zero.'''
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0),
                                    3 * np.exp(1j * np.pi)]
    expected_imaginary_coherence = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.imaginary_coherence().squeeze(),
        expected_imaginary_coherence)
