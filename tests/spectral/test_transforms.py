import numpy as np
from pytest import mark

from src.spectral.transforms import (_add_trial_axis, _sliding_window,
                                     Multitaper)


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
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        time_halfbandwidth_product=time_halfbandwidth_product)
    assert m.n_tapers == expected_n_tapers


@mark.parametrize(
    'sampling_frequency, time_window_duration, expected_duration',
    [(1000, None, 0.1), (2000, None, 0.05), (1000, 0.1, 0.1)])
def test_time_window_duration(sampling_frequency, time_window_duration,
                              expected_duration):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration)
    assert m.time_window_duration == expected_duration


@mark.parametrize(
    'sampling_frequency, time_window_step, expected_step',
    [(1000, None, 0.1), (2000, None, 0.05), (1000, 0.1, 0.1)])
def test_time_window_step(
        sampling_frequency, time_window_step, expected_step):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_step=time_window_step)
    assert m.time_window_step == expected_step


@mark.parametrize(
    'sampling_frequency, time_window_duration, expected_n_time_samples',
    [(1000, None, 100), (1000, 0.1, 100), (2000, 0.025, 50)])
def test_n_time_samples(
        sampling_frequency, time_window_duration, expected_n_time_samples):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration)
    assert m.n_time_samples == expected_n_time_samples


@mark.parametrize(
    ('sampling_frequency, time_window_duration, n_fft_samples,'
     'expected_n_fft_samples'),
    [(1000, None, 5, 5), (1000, 0.1, None, 100)])
def test_n_fft_samples(
    sampling_frequency, time_window_duration, n_fft_samples,
        expected_n_fft_samples):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        n_fft_samples=n_fft_samples)
    assert m.n_fft_samples == expected_n_fft_samples


def test_frequencies():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    n_fft_samples = 4
    sampling_frequency = 1000
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        n_fft_samples=n_fft_samples)
    expected_frequencies = np.array([0, 250, -500, -250])
    assert np.allclose(m.frequencies, expected_frequencies)


def test_n_signals():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series=time_series)
    assert m.n_signals == n_signals


def test_n_trials():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series=time_series)
    assert m.n_trials == n_trials

    time_series = np.zeros((n_time_samples, n_signals))
    m = Multitaper(time_series=time_series)
    assert m.n_trials == 1


@mark.parametrize(
    ('time_halfbandwidth_product, time_window_duration, '
     'expected_frequency_resolution'),
    [(3, .10, 30), (1, 0.02, 50), (5, 1, 5)])
def test_frequency_resolution(
        time_halfbandwidth_product, time_window_duration,
        expected_frequency_resolution):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        time_halfbandwidth_product=time_halfbandwidth_product,
        time_window_duration=time_window_duration)
    assert m.frequency_resolution == expected_frequency_resolution


@mark.parametrize(
    ('time_window_step, n_samples_per_time_step, '
     'expected_n_samples_per_time_step'),
    [(None, None, 100), (0.001, None, 1), (0.002, None, 2),
     (None, 10, 10)])
def test_n_samples_per_time_step(
        time_window_step, n_samples_per_time_step,
        expected_n_samples_per_time_step):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))

    m = Multitaper(
            time_window_duration=0.10,
            n_samples_per_time_step=n_samples_per_time_step,
            time_series=time_series,
            time_window_step=time_window_step)
    assert m.n_samples_per_time_step == expected_n_samples_per_time_step


@mark.parametrize('time_window_duration', [0.1, 0.2, 2.4, 0.16])
def test_time(time_window_duration):
    sampling_frequency = 1500
    start_time, end_time = -2.4, 2.4
    n_trials, n_signals = 10, 2
    n_time_samples = int(
        (end_time - start_time) * sampling_frequency) + 1
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)
    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]
    m = Multitaper(
        sampling_frequency=sampling_frequency,
        time_series=time_series,
        start_time=start_time,
        time_window_duration=time_window_duration)
    assert np.allclose(m.time, expected_time)


def test_tapers():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series, is_low_bias=False)
    assert np.allclose(m.tapers.shape, (n_time_samples, m.n_tapers))

    m = Multitaper(time_series, tapers=np.zeros((10, 3)))
    assert np.allclose(m.tapers.shape, (10, 3))
