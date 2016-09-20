import pytest
import numpy as np
import src.spectral as spectral


@pytest.mark.parametrize("sampling_frequency, number_of_fft_samples, desired_frequencies, \
                          expected_frequencies, expected_ind", [
    (1000, 10, None, [0, 100, 200, 300, 400, 500], [0, 1, 2, 3, 4, 5]),
    (1500, 10, None, [0, 150, 300, 450, 600, 750], [0, 1, 2, 3, 4, 5]),
    (1500, 10, [100, 200], [150], [1]),
    (1500, 8, None, [0, 187.5, 375, 562.5, 750], [0, 1, 2, 3, 4]),
    (1500, 8, [200, 600], [375, 562.5], [2, 3]),
])
def test_get_frequencies(sampling_frequency, number_of_fft_samples, desired_frequencies,
                         expected_frequencies, expected_ind):

    test_frequencies, test_ind = spectral._get_frequencies(sampling_frequency, number_of_fft_samples,
                                                           desired_frequencies=desired_frequencies)
    assert np.all(test_frequencies == expected_frequencies)
    assert np.all(test_ind == expected_ind)


@pytest.mark.parametrize("time_series_length, number_of_tapers, expected_shape", [
    (23, 1, (23, 1)),
    (23, 5, (23, 5)),
    (8, 3, (8, 3)),
])
def test_get_tapers_shape(time_series_length, number_of_tapers, expected_shape):
    SAMPLING_FREQUENCY = 1000
    TIME_HALFBANDWIDTH_PRODUCT = 3
    tapers = spectral._get_tapers(time_series_length, SAMPLING_FREQUENCY, TIME_HALFBANDWIDTH_PRODUCT,
                                  number_of_tapers)
    assert np.all(tapers.shape == expected_shape)

MEAN = 0
STD_DEV = 2


@pytest.mark.parametrize("time_series_length, number_of_tapers, data, number_of_fft_samples, \
                         expected_shape", [
    (23, 1, np.random.normal(MEAN, STD_DEV, (23, 1)), 8, (8, 1, 1)),
    (23, 3, np.random.normal(MEAN, STD_DEV, (23, 1)), 8, (8, 1, 3)),
    (23, 1, np.random.normal(MEAN, STD_DEV, (23, 2)), 8, (8, 2, 1)),
    (12, 1, np.random.normal(MEAN, STD_DEV, (12, 2)), 8, (8, 2, 1)),
    (12, 1, np.random.normal(MEAN, STD_DEV, 12), 8, (8, 1, 1)),
])
def test_multitaper_fft_shape(time_series_length, number_of_tapers, data, number_of_fft_samples,
                              expected_shape):
    SAMPLING_FREQUENCY = 1000
    TIME_HALFBANDWIDTH_PRODUCT = 3
    tapers = spectral._get_tapers(time_series_length, SAMPLING_FREQUENCY, TIME_HALFBANDWIDTH_PRODUCT,
                                  number_of_tapers)
    dft = spectral._multitaper_fft(tapers, data, number_of_fft_samples, SAMPLING_FREQUENCY)
    assert np.all(dft.shape == expected_shape)


@pytest.mark.parametrize("test_number, expected_number", [
    (3, 2),
    (17, 5),
    (1, 0),
])
def test_nextpower2(test_number, expected_number):
    assert spectral._nextpower2(test_number) == expected_number


@pytest.mark.parametrize("number_of_tapers, data, time_halfbandwidth_product, \
                          pad, expected_spectrum_shape, expected_freq_shape", [
    (None, np.random.normal(MEAN, STD_DEV, (23, 1)), 3, 0, (32, 1, 5), (17,)),
    (None, np.random.normal(MEAN, STD_DEV, (23, 1)), 1, 0, (32, 1, 1), (17,)),
    (2, np.random.normal(MEAN, STD_DEV, (23, 1)), 1, 0, (32, 1, 2), (17,)),
    (2, np.random.normal(MEAN, STD_DEV, (23, 2)), 1, 0, (32, 2, 2), (17,)),
    (2, np.random.normal(MEAN, STD_DEV, (15, 2)), 1, 0, (16, 2, 2), (9,)),
])
def test_multitaper_spectrum_shape(number_of_tapers, data, time_halfbandwidth_product, pad,
                                   expected_spectrum_shape, expected_freq_shape):
    SAMPLING_FREQUENCY = 1000
    complex_spectrum, frequencies, freq_ind = spectral.multitaper_spectrum(data, SAMPLING_FREQUENCY,
                                                                           time_halfbandwidth_product=time_halfbandwidth_product,
                                                                           number_of_tapers=number_of_tapers,
                                                                           pad=pad)
    assert np.all(complex_spectrum.shape == expected_spectrum_shape)
    assert np.all(freq_ind.shape == expected_freq_shape)


@pytest.mark.parametrize("complex_spectrum, expected_shape", [
    (np.random.normal(MEAN, STD_DEV, (23, 2, 1)), (23,)),
])
def test_cross_spectrum_shape(complex_spectrum, expected_shape):
    cross_spectrum = spectral._cross_spectrum(complex_spectrum, complex_spectrum)
    assert np.all(cross_spectrum.shape == expected_shape)
