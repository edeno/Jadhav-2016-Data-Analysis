import numpy as np
import pandas as pd
import pytest

from src.spectral import (_cross_spectrum, _get_frequencies, _get_tapers,
                          _get_window_lengths,
                          _make_sliding_window_dataframe,
                          _multitaper_fft, _nextpower2,
                          _get_multitaper_bias,
                          filter_significant_groups_less_than_frequency_resolution)


@pytest.mark.parametrize(
    'sampling_frequency, number_of_fft_samples, desired_frequencies, \
     expected_frequencies, expected_ind', [
        (1000, 10, None, [0, 100, 200, 300, 400, 500], [0, 1, 2, 3, 4, 5]),
        (1500, 10, None, [0, 150, 300, 450, 600, 750], [0, 1, 2, 3, 4, 5]),
        (1500, 10, [100, 200], [150], [1]),
        (1500, 8, None, [0, 187.5, 375, 562.5, 750], [0, 1, 2, 3, 4]),
        (1500, 8, [200, 600], [375, 562.5], [2, 3]),
    ])
def test_get_frequencies(sampling_frequency, number_of_fft_samples,
                         desired_frequencies, expected_frequencies,
                         expected_ind):
    test_frequencies, test_ind = _get_frequencies(
        sampling_frequency, number_of_fft_samples,
        desired_frequencies=desired_frequencies)
    assert np.all(test_frequencies == expected_frequencies)
    assert np.all(test_ind == expected_ind)


@pytest.mark.parametrize(
    'time_series_length, number_of_tapers, expected_shape', [
        (23, 1, (23, 1)),
        (23, 5, (23, 5)),
        (8, 3, (8, 3)),
    ])
def test_get_tapers_shape(time_series_length, number_of_tapers,
                          expected_shape):
    SAMPLING_FREQUENCY = 1000
    TIME_HALFBANDWIDTH_PRODUCT = 3
    tapers = _get_tapers(
        time_series_length, SAMPLING_FREQUENCY, TIME_HALFBANDWIDTH_PRODUCT,
        number_of_tapers)
    assert np.all(tapers.shape == expected_shape)

MEAN = 0
STD_DEV = 2


@pytest.mark.parametrize(
    'time_series_length, number_of_tapers, data, number_of_fft_samples, \
     expected_shape', [
        (23, 1, np.random.normal(MEAN, STD_DEV, (23, 1)), 8, (8, 1, 1)),
        (23, 3, np.random.normal(MEAN, STD_DEV, (23, 1)), 8, (8, 1, 3)),
        (23, 1, np.random.normal(MEAN, STD_DEV, (23, 2)), 8, (8, 2, 1)),
        (12, 1, np.random.normal(MEAN, STD_DEV, (12, 2)), 8, (8, 2, 1)),
        (12, 1, np.random.normal(MEAN, STD_DEV, 12), 8, (8, 1, 1)),
    ])
def test_multitaper_fft_shape(time_series_length, number_of_tapers, data,
                              number_of_fft_samples, expected_shape):
    SAMPLING_FREQUENCY = 1000
    TIME_HALFBANDWIDTH_PRODUCT = 3
    tapers = _get_tapers(
        time_series_length, SAMPLING_FREQUENCY, TIME_HALFBANDWIDTH_PRODUCT,
        number_of_tapers)
    dft = _multitaper_fft(
        tapers, data, number_of_fft_samples, SAMPLING_FREQUENCY)
    assert np.all(dft.shape == expected_shape)


@pytest.mark.parametrize('test_number, expected_number', [
    (3, 2),
    (17, 5),
    (1, 0),
])
def test_nextpower2(test_number, expected_number):
    assert _nextpower2(test_number) == expected_number


@pytest.mark.parametrize('complex_spectrum, expected_shape', [
    (np.random.normal(MEAN, STD_DEV, (23, 2, 1)), (23, 2, 1)),
])
def test_cross_spectrum_shape(complex_spectrum, expected_shape):
    cross_spectrum = _cross_spectrum(
        complex_spectrum, complex_spectrum)
    assert np.all(cross_spectrum.shape == expected_shape)


@pytest.mark.parametrize(
    'num_data, time_window_duration, time_window_step, sampling_frequency',
    [(1000, 0.1, 0.1, 1000), (1000, 0.1, 0.05, 1000),
     (1000, 0.1, 0.3, 1000), (1000, 0.1, 0.3, 1500),
     (1500, 0.1, 0.3, 1500)])
def test_make_sliding_window_dataframe(num_data, time_window_duration,
                                       time_window_step,
                                       sampling_frequency):
    data = np.ones((num_data, 1))
    time = np.linspace(0, num_data / sampling_frequency,
                       num=num_data, endpoint=False)
    time_step_length, time_window_length = _get_window_lengths(
        time_window_duration, sampling_frequency, time_window_step)
    axis = 0

    def test_func(data, **kwargs):
        return pd.DataFrame(kwargs)

    kwargs = {'test1': [1, 2]}
    dataframes = list(_make_sliding_window_dataframe(
        test_func, [data], time_window_duration, time_window_step,
        time_step_length, time_window_length, time, axis, **kwargs))
    expected_time_steps = np.arange(
        time_window_duration / 2, time[-1] + 1 / sampling_frequency,
        time_window_step)
    if (time_step_length * (len(expected_time_steps) - 1) +
            time_window_length > num_data):
        expected_time_steps = expected_time_steps[:-1]

    assert len(dataframes) == len(expected_time_steps)
    assert np.all([df.test1.values == [1, 2] for df in dataframes])
    assert np.allclose(
        [df.index.values[0][1] for df in dataframes], expected_time_steps)


def test__get_multitaper_bias():
    n_trials, n_tapers = 10, 5
    bias = _get_multitaper_bias(n_trials, n_tapers)
    expected = 1 / 98
    assert bias == expected


def _convert_to_significant_series(x):
    FREQUENCIES = pd.Index(np.arange(0, 20, 2), name='frequency')
    return pd.Series(
        x, index=FREQUENCIES, name='is_significant').astype(bool)


@pytest.mark.parametrize(
    'is_significant, frequency_resolution, expected', [
        (_convert_to_significant_series(np.zeros((10,))), 2,
         _convert_to_significant_series(np.zeros((10,)))),
        (_convert_to_significant_series(np.ones((10,))), 20,
         _convert_to_significant_series(np.zeros((10,)))),
        (_convert_to_significant_series(np.ones((10,))), 1,
         _convert_to_significant_series(np.ones((10,)))),
        (_convert_to_significant_series(np.ones((10,))), 3,
         _convert_to_significant_series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])),
        (_convert_to_significant_series([0, 0, 0, 0, 0, 1, 1, 1, 0, 0]), 1,
         _convert_to_significant_series([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])),
        (_convert_to_significant_series([0, 0, 0, 0, 0, 1, 1, 1, 0, 0]), 3,
         _convert_to_significant_series([0, 0, 0, 0, 0, 1, 0, 1, 0, 0])),
        (_convert_to_significant_series([0, 0, 0, 0, 0, 1, 0, 1, 0, 0]), 3,
         _convert_to_significant_series(np.zeros((10,)))),
        (_convert_to_significant_series([1, 1, 1, 0, 0, 1, 1, 1, 0, 0]), 3,
         _convert_to_significant_series([1, 0, 1, 0, 0, 1, 0, 1, 0, 0])),
        (_convert_to_significant_series([0, 0, 0, 0, 0, 1, 1, 1, 0, 0]), 5,
         _convert_to_significant_series(np.zeros((10,)))),
    ])
def test_filter_significant_groups_less_than_frequency_resolution(
        is_significant, frequency_resolution, expected):
    is_sig = filter_significant_groups_less_than_frequency_resolution(
        is_significant, frequency_resolution)
    assert all(is_sig == expected)
