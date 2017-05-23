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
                                    3 * np.exp(1j * 0)]
    expected_imaginary_coherence = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.imaginary_coherence().squeeze(),
        expected_imaginary_coherence)


def test_phase_locking_value():
    '''Make sure phase locking value ignores magnitudes.'''
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = (
        np.random.uniform(0, 2, (n_time_samples, n_trials, n_tapers,
                                 n_fft_samples, n_signals)) *
        np.exp(1j * np.pi / 2))
    expected_phase_locking_value_magnitude = np.ones(
        fourier_coefficients.shape)
    expected_phase_locking_value_angle = np.zeros(
        fourier_coefficients.shape)
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.allclose(
        np.abs(this_Conn.phase_locking_value()),
        expected_phase_locking_value_magnitude)
    assert np.allclose(
        np.angle(this_Conn.phase_locking_value()),
        expected_phase_locking_value_angle)


def test_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0),
                                    3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.phase_lag_index().squeeze(),
        expected_phase_lag_index)


def test_phase_lag_index_sets_angles_up_to_pi_to_same_value():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)
    fourier_coefficients[..., 0] = (np.random.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)) *
        np.exp(1j * np.pi / 2))
    fourier_coefficients[..., 1] = (np.random.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)) *
        np.exp(1j * np.pi / 4))

    expected_phase_lag_index = np.zeros((2, 2))
    expected_phase_lag_index[0, 1] = 1
    expected_phase_lag_index[1, 0] = -1

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.allclose(
        this_Conn.phase_lag_index().squeeze(),
        expected_phase_lag_index)


def test_weighted_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0),
                                    3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.weighted_phase_lag_index().squeeze(),
        expected_phase_lag_index)


def test_weighted_phase_lag_index_is_same_as_phase_lag_index():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [1 * np.exp(1j * 3 * np.pi / 4),
                                    1 * np.exp(1j * 5 * np.pi / 4)]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.phase_lag_index(),
        this_Conn.weighted_phase_lag_index())


def test_debiased_squared_phase_lag_index():
    '''Test that incoherent signals are set to zero or below.'''
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles2 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))

    fourier_coefficients[..., 0] = np.exp(1j * angles1)
    fourier_coefficients[..., 1] = np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.all(
        this_Conn.debiased_squared_phase_lag_index() < np.finfo(float).eps)


def test_debiased_squared_weighted_phase_lag_index():
    '''Test that incoherent signals are set to zero or below.'''
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles2 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))

    fourier_coefficients[..., 0] = np.exp(1j * angles1)
    fourier_coefficients[..., 1] = np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    # set NaN to 0 so less than will work
    debiased_wPLI = this_Conn.debiased_squared_weighted_phase_lag_index()
    debiased_wPLI[np.isnan(debiased_wPLI)] = 0

    assert np.all(debiased_wPLI < np.finfo(float).eps)


def test_pairwise_phase_consistency():
    '''Test that incoherent signals are set to zero or below
    and that differences in power are ignored.'''
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    magnitude1 = np.random.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    magnitude2 = np.random.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles2 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))

    fourier_coefficients[..., 0] = magnitude1 * np.exp(1j * angles1)
    fourier_coefficients[..., 1] = magnitude2 * np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    ppc = this_Conn.pairwise_phase_consistency()

    # set diagonal to zero because its always 1
    diagonal_ind = np.arange(0, n_signals)
    ppc[..., diagonal_ind, diagonal_ind] = 0

    assert np.all(ppc < np.finfo(float).eps)
