import numpy as np
from pytest import mark

from src.spectral.transforms import (_add_trial_axis, _sliding_window,
                                     _nextpower2, Multitaper)


def test__add_trial_axis():
    n_time_samples, n_signals = (2, 3)
    test_data = np.ones(
        (n_time_samples, n_signals))
    expected_shape = (n_time_samples, 1, n_signals)
    assert np.allclose(_add_trial_axis(test_data).shape, expected_shape)

    # if there is a trial dimension, do nothing
    n_trials = 10
    test_data = np.ones(
        (n_time_samples, n_trials, n_signals))
    expected_shape = (n_time_samples, n_trials, n_signals)
    assert np.allclose(_add_trial_axis(test_data).shape, expected_shape)


@mark.parametrize('test_number, expected_number', [
    (3, 2),
    (17, 5),
    (1, 0)])
def test__nextpower2(test_number, expected_number):
    assert _nextpower2(test_number) == expected_number


@mark.parametrize(
    'test_array, window_size, step_size, axis, expected_array',
    [(np.arange(1, 6), 3, 1, -1, np.array([[1, 2, 3],
                                           [2, 3, 4],
                                           [3, 4, 5]])),
     (np.arange(1, 6), 3, 2, -1, np.array([[1, 2, 3],
                                           [3, 4, 5]])),
     (np.arange(0, 6).reshape((2, 3)), 2, 1, 0, np.array([[[0, 3],
                                                           [1, 4],
                                                           [2, 5]]]))
     ])
def test__sliding_window(
        test_array, window_size, step_size, axis, expected_array):
    assert np.allclose(
        _sliding_window(
            test_array, window_size=window_size, step_size=step_size,
            axis=axis),
        expected_array)


@mark.parametrize(
    'time_halfbandwidth_product, expected_n_tapers',
    [(3, 5), (1, 1), (1.75, 2)])
def test_n_tapers(time_halfbandwidth_product, expected_n_tapers):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros(
        (n_time_samples, n_trials, n_signals), dtype=np.complex)
    m = Multitaper(
        time_series=time_series,
        time_halfbandwidth_product=time_halfbandwidth_product)
    assert m.n_tapers == expected_n_tapers
