import numpy as np
from pytest import mark

from src.spectral.transforms import (_add_trial_axis, _sliding_window,
                                     _nextpower2)


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
