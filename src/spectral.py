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


def make_timefrequency_dataframe(lfp_data_frame, time_window_duration, time_window_step, frequency_resolution, sampling_frequency):
    ''' Generator function that returns a power spectral density data frame for each time window
    '''
    time_window_start = 0
    number_points_time_window = int(np.fix(time_window_duration * sampling_frequency))
    number_points_time_step = int(np.fix(time_window_step * sampling_frequency))

    while time_window_start + number_points_time_window < len(lfp_data_frame):
        try:
            time_window_end = time_window_start + number_points_time_window
            windowed_data_frame = lfp_data_frame.iloc[time_window_start:time_window_end, :]
            centered_signal = (windowed_data_frame['electric_potential'] -
                               windowed_data_frame['electric_potential'].mean())
            frequencies, power_spectral_density, _ = tsa.multi_taper_psd(centered_signal,
                                                                         BW=frequency_resolution,
                                                                         Fs=sampling_frequency,
                                                                         adaptive=False,
                                                                         jackknife=False
                                                                         )
            centered_time = (lfp_data_frame.index.values[time_window_start] +
                             (time_window_duration / 2))
            centered_time = np.round(centered_time, decimals=4)

            yield (pd.DataFrame({'power': power_spectral_density,
                                'frequency': np.round(frequencies, decimals=3)}
                                )
                   .assign(time=centered_time)
                   )
            time_window_start += number_points_time_step
        except ValueError:
            # Not enough data points
            pass


def get_spectrogram_dataframe(lfp_data_frame,
                              sampling_frequency=1000,
                              frequency_resolution=2,
                              time_window_duration=1,
                              time_window_step=0):
    ''' Returns a pandas dataframe with the information for a spectrogram.
    Sampling frequency and frequency resolution inputs are given in Hertz.
    Time window duration and steps are given in seconds.
    '''
    if time_window_step == 0:
        time_window_step = time_window_duration
    return pd.concat(
            (time_window for time_window in make_timefrequency_dataframe(lfp_data_frame,
                                                                         time_window_duration,
                                                                         time_window_step,
                                                                         frequency_resolution,
                                                                         sampling_frequency))
            )


def _get_time_freq_from_spectrogram(spectrogram_dataframe):
    return (np.unique(spectrogram_dataframe['time']), np.unique(spectrogram_dataframe['frequency']))

def plot_spectrogram(spectrogram_dataframe, axis_handle):
    time, freq = _get_time_freq_from_spectrogram(spectrogram_dataframe)
    mesh = axis_handle.pcolormesh(time, freq, spectrogram_dataframe.pivot('frequency', 'time', 'power'),
                                  cmap='viridis',
                                  shading='gouraud')
    axis_handle.set_ylabel('Frequency (Hz)')
    axis_handle.set_xlabel('Time (seconds)')
    axis_handle.set_xlim([time.min(), time.max()])
    axis_handle.set_ylim([freq.min(), freq.max()])
    return mesh
