# -*- coding: utf-8 -*-

import scipy.io
import scipy.fftpack
import scipy.signal
import numpy as np
import nitime.algorithms as tsa
import matplotlib.pyplot as plt
import pandas as pd


def plot_spectrum_nitime(data_frame, sampling_frequency, axis_handle):
    ''' Plots the spectrum of a pandas dataframe containing the LFP time domain
    signal ('electric_potential') using the nitime package. Spectrum is computed
    using the fft and a rectangular window.
    '''
    centered_signal = data_frame['electric_potential'] - data_frame['electric_potential'].mean()
    frequencies, power_spectral_density = tsa.periodogram(centered_signal,
                                                          Fs=sampling_frequency,
                                                          normalize=True)
    axis_handle.plot(frequencies, power_spectral_density)


def tetrode_title(tetrode_index_tuple, cur_tetrode_info):
    ''' Returns a string that identifies the tetrode in a semantic way
    '''
    return ('{brain_area} Tetrode #{tetrode_number}'
            .format(tetrode_number=tetrode_index_tuple[-1],
                    brain_area=cur_tetrode_info.loc[tetrode_index_tuple, 'area']))


def plot_session(data, plotting_function):
    num_rows = int(np.ceil(len(data) / 7))
    fig, axis_handles = plt.subplots(num_rows, 7,
                                     figsize=(21, 9),
                                     sharex=True,
                                     sharey=True)
    for ind, axis_handle in enumerate(axis_handles.flatten()):
        try:
            plotting_function(data[ind], axis_handle)
        except IndexError:
            pass


def make_windowed_spectrum_dataframe(lfp_dataframe, time_window_duration, time_window_step,
                                     sampling_frequency, desired_frequencies=None,
                                     time_halfbandwidth_product=3, number_of_tapers=None, pad=0,
                                     tapers=None):
    ''' Generator function that returns a power spectral density data frame for each time window
    '''
    time_window_start = 0
    number_points_time_window = int(np.fix(time_window_duration * sampling_frequency))
    number_points_time_step = int(np.fix(time_window_step * sampling_frequency))

    while time_window_start + number_points_time_window < len(lfp_dataframe):
        try:
            time_window_end = time_window_start + number_points_time_window
            windowed_dataframe = lfp_dataframe.iloc[time_window_start:time_window_end, :]
            centered_signal = np.array(windowed_dataframe['electric_potential'] -
                                       windowed_dataframe['electric_potential'].mean())
            power_spectral_density, frequencies = multitaper_power_spectral_density(centered_signal,
                                                                                    sampling_frequency,
                                                                                    desired_frequencies=desired_frequencies,
                                                                                    time_halfbandwidth_product=time_halfbandwidth_product,
                                                                                    number_of_tapers=number_of_tapers,
                                                                                    pad=pad,
                                                                                    tapers=tapers)
            centered_time = lfp_dataframe.index.values[time_window_start] + \
                (time_window_duration / 2)

            yield (pd.DataFrame({'power': power_spectral_density,
                                 'frequency': frequencies}
                                )
                   .assign(time=centered_time)
                   )
            time_window_start += number_points_time_step
        except ValueError:
            # Not enough data points
            pass


def get_spectrogram_dataframe(lfp_dataframe,
                              sampling_frequency=1000,
                              time_window_duration=1,
                              time_window_step=None,
                              desired_frequencies=None,
                              time_halfbandwidth_product=3,
                              number_of_tapers=None,
                              pad=0,
                              tapers=None):
    ''' Returns a pandas dataframe with the information for a spectrogram.
    Sampling frequency and frequency resolution inputs are given in Hertz.
    Time window duration and steps are given in seconds.
    '''
    if time_window_step is None:
        time_window_step = time_window_duration
    return pd.concat(
            (time_window for time_window in make_windowed_spectrum_dataframe(lfp_dataframe,
                                                                             time_window_duration,
                                                                             time_window_step,
                                                                             sampling_frequency,
                                                                             desired_frequencies=desired_frequencies,
                                                                             time_halfbandwidth_product=time_halfbandwidth_product,
                                                                             number_of_tapers=number_of_tapers,
                                                                             pad=pad,
                                                                             tapers=tapers))
            )


def _get_time_freq_from_spectrogram(spectrogram_dataframe):
    return (np.unique(spectrogram_dataframe['time']),
            np.unique(spectrogram_dataframe['frequency']))


def plot_spectrogram(spectrogram_dataframe, axis_handle, spectrum_name='power', cmap='viridis'):
    time, freq = _get_time_freq_from_spectrogram(spectrogram_dataframe)
    mesh = axis_handle.pcolormesh(time, freq, spectrogram_dataframe.pivot('frequency', 'time', spectrum_name),
                                  cmap=cmap,
                                  shading='gouraud',
                                  vmin=spectrogram_dataframe[spectrum_name].quantile(q=0.05),
                                  vmax=spectrogram_dataframe[spectrum_name].quantile(q=0.95))
    axis_handle.set_ylabel('Frequency (Hz)')
    axis_handle.set_xlabel('Time (seconds)')
    axis_handle.set_xlim([time.min(), time.max()])
    axis_handle.set_ylim([freq.min(), freq.max()])
    return mesh


def _get_frequencies(sampling_frequency, number_of_fft_samples, desired_frequencies=None):
    ''' Returns the frequencies and frequency index. Desired frequencies is a two element
    tuple that defines the range of frequencies returned. If unset, all frequencies from 0
    to the Nyquist (sampling_frequency / 2) are returned.
    '''
    if desired_frequencies is None:
        desired_frequencies = [0, sampling_frequency / 2]
    frequencies = np.linspace(0, sampling_frequency, num=number_of_fft_samples)
    frequencies = frequencies[:number_of_fft_samples]
    frequency_ind = np.where((desired_frequencies[0] <= frequencies) &
                             (frequencies <= desired_frequencies[1]))[0]
    return frequencies[frequency_ind], frequency_ind


def _get_tapers(time_series_length, sampling_frequency, time_halfbandwidth_product,
                number_of_tapers):
    ''' Returns the Discrete prolate spheroidal sequences (tapers) for multi-taper
    spectral analysis.
    '''
    tapers, _ = tsa.spectral.dpss_windows(time_series_length,
                                          time_halfbandwidth_product,
                                          number_of_tapers)
    return tapers.T * np.sqrt(sampling_frequency)


def _multitaper_fft(tapers, data, number_of_fft_samples, sampling_frequency):
    ''' Projects the data on the tapers and returns the discrete Fourier transform
    (frequencies x trials x tapers)
    '''
    try:
        projected_data = data[:, :, np.newaxis] * tapers[:, np.newaxis, :]
    except IndexError:
        # There are no trials
        projected_data = data[:, np.newaxis, np.newaxis] * tapers[:, :, np.newaxis]
    return scipy.fftpack.fft(projected_data, n=number_of_fft_samples, axis=0) / sampling_frequency


def _nextpower2(n):
    """Return the next integral power of two greater than the given number.
    Specifically, return m such that
        m >= n
        m == 2**x
    where x is an integer.
    This is useful for ensuring fast FFT sizes.
    """
    return int(2 ** np.ceil(np.log2(n)))


def multitaper_spectrum(data, sampling_frequency, desired_frequencies=None,
                        time_halfbandwidth_product=3, number_of_tapers=None, pad=0,
                        tapers=None):
    ''' Returns complex spectrum (frequencies x trials x tapers)
    '''
    if number_of_tapers is None:
        number_of_tapers = int(np.floor(2 * time_halfbandwidth_product - 1))
    if pad is None:
        pad = -1
    time_series_length = data.shape[0]
    number_of_fft_samples = max(_nextpower2(time_series_length) + pad, time_series_length)

    frequencies, freq_ind = _get_frequencies(sampling_frequency, number_of_fft_samples,
                                             desired_frequencies=desired_frequencies)
    if tapers is None:
        tapers = _get_tapers(time_series_length, sampling_frequency,
                             time_halfbandwidth_product, number_of_tapers)
    complex_spectrum = _multitaper_fft(tapers, data, number_of_fft_samples, sampling_frequency)
    return complex_spectrum, frequencies, freq_ind


def _cross_spectrum(complex_spectrum1, complex_spectrum2):
    '''Returns the cross-spectrum between two spectra'''
    cross_spectrum = np.conj(complex_spectrum1) * complex_spectrum2
    return np.mean(cross_spectrum, axis=(1, 2)).squeeze()


def multitaper_power_spectral_density(data, sampling_frequency, desired_frequencies=None,
                                      time_halfbandwidth_product=3, number_of_tapers=None,
                                      pad=0, tapers=None):
    ''' Returns the multi-taper power spectral density of a time series
    '''
    complex_spectrum, frequencies, freq_ind = multitaper_spectrum(data, sampling_frequency,
                                                                  desired_frequencies=desired_frequencies,
                                                                  time_halfbandwidth_product=time_halfbandwidth_product,
                                                                  number_of_tapers=number_of_tapers,
                                                                  tapers=tapers,
                                                                  pad=pad)
    psd = np.real(_cross_spectrum(complex_spectrum[freq_ind, :, :], complex_spectrum[freq_ind, :, :]))
    return psd, frequencies


def multitaper_coherency(data1, data2, sampling_frequency, desired_frequencies=None,
                         time_halfbandwidth_product=3, number_of_tapers=None, pad=0,
                         tapers=None):
    ''' Returns the multi-taper coherency of two time series
    data1 (time x trials)
    data2 (time x trials)
    '''
    complex_spectrum1, frequencies, freq_ind = multitaper_spectrum(data1, sampling_frequency,
                                                                   desired_frequencies=desired_frequencies,
                                                                   time_halfbandwidth_product=time_halfbandwidth_product,
                                                                   number_of_tapers=number_of_tapers,
                                                                   tapers=tapers,
                                                                   pad=pad)
    complex_spectrum2, _, _ = multitaper_spectrum(data2, sampling_frequency,
                                                  desired_frequencies=desired_frequencies,
                                                  time_halfbandwidth_product=time_halfbandwidth_product,
                                                  number_of_tapers=number_of_tapers,
                                                  tapers=tapers,
                                                  pad=pad)
    cross_spectrum = _cross_spectrum(complex_spectrum1[freq_ind, :, :], complex_spectrum2[freq_ind, :, :])
    spectrum1 = _cross_spectrum(complex_spectrum1[freq_ind, :, :], complex_spectrum1[freq_ind, :, :])
    spectrum2 = _cross_spectrum(complex_spectrum2[freq_ind, :, :], complex_spectrum2[freq_ind, :, :])

    coherency = cross_spectrum / np.sqrt(spectrum1 * spectrum2)
    return np.abs(coherency), np.angle(coherency), frequencies, np.real(spectrum1), np.real(spectrum2)


def make_windowed_coherency_dataframe(lfp_dataframe1, lfp_dataframe2, time_window_duration,
                                      time_window_step, sampling_frequency,
                                      desired_frequencies=None, time_halfbandwidth_product=3,
                                      number_of_tapers=None, pad=0, tapers=None):
    ''' Generator function that returns a coherency dataframe for each time window
    '''
    time_window_start = 0
    number_points_time_window = int(np.fix(time_window_duration * sampling_frequency))
    number_points_time_step = int(np.fix(time_window_step * sampling_frequency))

    while time_window_start + number_points_time_window < len(lfp_dataframe1):
        try:
            time_window_end = time_window_start + number_points_time_window
            windowed_dataframe1 = lfp_dataframe1.iloc[time_window_start:time_window_end, :]
            windowed_dataframe2 = lfp_dataframe2.iloc[time_window_start:time_window_end, :]
            centered_signal1 = np.array(windowed_dataframe1['electric_potential'] -
                                        windowed_dataframe1['electric_potential'].mean())
            centered_signal2 = np.array(windowed_dataframe2['electric_potential'] -
                                        windowed_dataframe2['electric_potential'].mean())
            coherence_magnitude, coherence_angle, frequencies, psd1, psd2 = \
                multitaper_coherency(centered_signal1,
                                     centered_signal2,
                                     sampling_frequency,
                                     desired_frequencies=desired_frequencies,
                                     time_halfbandwidth_product=time_halfbandwidth_product,
                                     number_of_tapers=number_of_tapers,
                                     pad=pad)
            centered_time = (lfp_dataframe1.index.values[time_window_start] +
                             (time_window_duration / 2))

            yield (pd.DataFrame({'frequency': frequencies,
                                 'coherence_magnitude': coherence_magnitude,
                                 'coherence_angle': coherence_angle,
                                 'power_spectrum1': psd1,
                                 'power_spectrum2': psd2
                                 })
                   .assign(time=centered_time)
                   )
            time_window_start += number_points_time_step
        except ValueError:
            # Not enough data points
            pass


def get_coherence_dataframe(lfp_dataframe1, lfp_dataframe2,
                            sampling_frequency=1000,
                            time_window_duration=1,
                            time_window_step=None,
                            desired_frequencies=None,
                            time_halfbandwidth_product=3,
                            number_of_tapers=None,
                            pad=0,
                            tapers=None):
    ''' Returns a pandas dataframe with the information for a spectrogram.
    Sampling frequency and frequency resolution inputs are given in Hertz.
    Time window duration and steps are given in seconds.
    '''
    if time_window_step is None:
        time_window_step = time_window_duration
    return pd.concat(
                    (time_window for time_window in make_windowed_coherency_dataframe(lfp_dataframe1,
                                                                                      lfp_dataframe2,
                                                                                      time_window_duration,
                                                                                      time_window_step,
                                                                                      sampling_frequency,
                                                                                      desired_frequencies=desired_frequencies,
                                                                                      time_halfbandwidth_product=time_halfbandwidth_product,
                                                                                      number_of_tapers=number_of_tapers,
                                                                                      pad=pad,
                                                                                      tapers=tapers)
                     )
                     )


