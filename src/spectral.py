# -*- coding: utf-8 -*-

import functools
import inspect
import scipy.io
import scipy.fftpack
import scipy.signal
import scipy.stats
import numpy as np
import nitime.algorithms as tsa
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import data_processing


def convert_pandas(func):
    is_time = 'time' in inspect.signature(func).parameters.keys()

    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            if is_time:
                try:
                    time = data.reset_index('time').time.values
                    kwargs['time'] = time
                except AttributeError:
                    raise AttributeError(
                        'No time column or index provided in dataframe')
            data = data.values
        elif isinstance(data, list) and isinstance(data[0], pd.DataFrame):
            if is_time:
                try:
                    time = data[0].reset_index('time').time.values
                    kwargs['time'] = time
                except AttributeError:
                    raise AttributeError(
                        'No time column or index provided in dataframe')
            data = [datum.values for datum in data]

        return func(data, *args, **kwargs)
    return wrapper


def tetrode_title(tetrode_index_tuple, cur_tetrode_info):
    ''' Returns a string that identifies the tetrode in a semantic way
    '''
    return ('{brain_area} Tetrode #{tetrode_number}'
            .format(tetrode_number=tetrode_index_tuple[-1],
                    brain_area=cur_tetrode_info.loc[tetrode_index_tuple, 'area']))


def _center_data(x):
    '''Returns the mean-centered data array along the first axis'''
    return x - np.nanmean(x, axis=0)


def _get_window_array(data, time_window_start_ind, time_window_end_ind):
    '''Returns the data for a given start and end index'''
    window_array = [datum[time_window_start_ind:time_window_end_ind, ...]
                    for datum in data]
    if len(window_array) == 1:
        window_array = window_array[0]
    return window_array


def _make_sliding_window_dataframe(func, data, time_window_duration, time_window_step,
                                   time_step_length, time_window_length, time,
                                   **kwargs):
    ''' Generator function that returns a transformed dataframe (via func) for each sliding
    time window.
    '''
    time_window_start_ind = 0

    while time_window_start_ind + time_window_length <= len(data[0]):
        try:
            time_window_end_ind = time_window_start_ind + time_window_length
            windowed_arrays = _get_window_array(
                data, time_window_start_ind, time_window_end_ind)

            yield (func(windowed_arrays, **kwargs)
                   .assign(time=_get_window_center(time_window_start_ind, time_window_duration,
                                                   time))
                   .set_index('time', append=True))
            time_window_start_ind += time_step_length
        except ValueError:
            # Not enough data points
            raise StopIteration


@convert_pandas
def multitaper_spectrogram(data,
                           sampling_frequency=1000,
                           time_halfbandwidth_product=3,
                           time_window_duration=1,
                           pad=0,
                           time_window_step=None,
                           desired_frequencies=None,
                           number_of_tapers=None,
                           tapers=None,
                           time=None,
                           number_of_fft_samples=None):
    ''' Returns a pandas dataframe with columns time, frequency, power


    '''
    time_step_length, time_window_length = _get_window_lengths(
        time_window_duration,
        sampling_frequency,
        time_window_step)
    tapers, number_of_fft_samples, frequencies, freq_ind = _set_default_multitaper_parameters(
        number_of_time_samples=time_window_length,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        time_window_step=time_window_duration,
        tapers=tapers,
        number_of_tapers=number_of_tapers,
        time_halfbandwidth_product=time_halfbandwidth_product,
        desired_frequencies=desired_frequencies,
        pad=pad)
    return pd.concat(list(_make_sliding_window_dataframe(
        multitaper_power_spectral_density,
        [data],
        time_window_duration,
        time_window_step,
        time_step_length,
        time_window_length,
        time,
        sampling_frequency=sampling_frequency,
        desired_frequencies=desired_frequencies,
        time_halfbandwidth_product=time_halfbandwidth_product,
        number_of_tapers=number_of_tapers,
        pad=pad,
        tapers=tapers,
        frequencies=frequencies,
        freq_ind=freq_ind,
        number_of_fft_samples=number_of_fft_samples)).sort_index()
    )


def _set_default_multitaper_parameters(number_of_time_samples=None, sampling_frequency=None,
                                       time_window_step=None, time_window_duration=None,
                                       tapers=None, number_of_tapers=None,
                                       time_halfbandwidth_product=None, pad=None,
                                       number_of_fft_samples=None, frequencies=None,
                                       freq_ind=None, desired_frequencies=None):
    '''Function to help set default multitaper parameters given that some subset of them are
     unset
     '''
    if tapers is None:
        if number_of_tapers is None:
            number_of_tapers = int(
                np.floor(2 * time_halfbandwidth_product - 1))

        tapers = _get_tapers(number_of_time_samples,
                             sampling_frequency,
                             time_halfbandwidth_product,
                             number_of_tapers)
    if pad is None:
        pad = -1
    if number_of_fft_samples is None:
        next_exponent = _nextpower2(number_of_time_samples)
        number_of_fft_samples = max(
            2 ** (next_exponent + pad), number_of_time_samples)
    if frequencies is None:
        frequencies, freq_ind = _get_frequencies(sampling_frequency,
                                                 number_of_fft_samples,
                                                 desired_frequencies=desired_frequencies)
    return tapers, number_of_fft_samples, frequencies, freq_ind


def _get_window_lengths(time_window_duration, sampling_frequency, time_window_step):
    '''Figures out the number of points per time window and step'''
    time_window_length = int(np.fix(time_window_duration * sampling_frequency))
    if time_window_step is None:
        time_window_step = time_window_duration
    time_step_length = int(np.fix(time_window_step * sampling_frequency))
    return time_step_length, time_window_length


def _get_unique_time_freq(spectrogram_dataframe):
    '''Returns the unique time and frequency given a spectrogram dataframe with
    non-unique time and frequency columns'''
    return (np.unique(spectrogram_dataframe.reset_index().time.values),
            np.unique(spectrogram_dataframe.reset_index().frequency.values))


def plot_spectrogram(spectrogram_dataframe, axis_handle=None,
                     spectrum_name='power', cmap='viridis',
                     time_units='seconds', frequency_units='Hz',
                     vmin=None, vmax=None, plot_type=None):
    if axis_handle is None:
        axis_handle = plt.gca()
    if vmin is None:
        vmin = spectrogram_dataframe[spectrum_name].quantile(q=0.05)
    if vmax is None:
        vmax = spectrogram_dataframe[spectrum_name].quantile(q=0.95)
    time, freq = _get_unique_time_freq(spectrogram_dataframe)
    data2D = spectrogram_dataframe.reset_index().pivot(
        'frequency', 'time', spectrum_name)
    if plot_type is None:
        mesh = axis_handle.pcolormesh(time, freq, data2D,
                                      cmap=cmap,
                                      shading='gouraud',
                                      vmin=vmin,
                                      vmax=vmax)
    elif plot_type is 'change':
        mesh = axis_handle.pcolormesh(time, freq, data2D,
                                      cmap=cmap,
                                      shading='gouraud',
                                      norm=colors.LogNorm(vmin=vmin, vmax=vmax))

    axis_handle.set_ylabel('Frequency ({frequency_units})'.format(
        frequency_units=frequency_units))
    axis_handle.set_xlabel('Time ({time_units})'.format(time_units=time_units))
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
    frequencies = np.linspace(0, sampling_frequency,
                              num=number_of_fft_samples + 1)
    frequency_ind = np.where((desired_frequencies[0] <= frequencies) &
                             (frequencies <= desired_frequencies[1]))[0]
    return frequencies[frequency_ind], frequency_ind


def _get_tapers(time_series_length, sampling_frequency, time_halfbandwidth_product,
                number_of_tapers):
    ''' Returns the Discrete prolate spheroidal sequences (tapers) for multi-taper
    spectral analysis (time series length x tapers).
    '''
    tapers, _ = tsa.spectral.dpss_windows(time_series_length,
                                          time_halfbandwidth_product,
                                          number_of_tapers)
    return tapers.T * np.sqrt(sampling_frequency)


def _multitaper_fft(tapers, data, number_of_fft_samples, sampling_frequency):
    ''' Projects the data on the tapers and returns the discrete Fourier transform
    (frequencies x trials x tapers) with len(frequencies) = number of fft samples
    '''
    try:
        projected_data = data[:, :, np.newaxis] * tapers[:, np.newaxis, :]
    except IndexError:
        # There are no trials
        projected_data = data[:, np.newaxis,
                              np.newaxis] * tapers[:, :, np.newaxis]
    return scipy.fftpack.fft(projected_data, n=number_of_fft_samples, axis=0) / sampling_frequency


def _nextpower2(n):
    """Return the next integer exponent of two greater than the given number.
    This is useful for ensuring fast FFT sizes.
    """
    return int(np.ceil(np.log2(n)))


def _cross_spectrum(complex_spectrum1, complex_spectrum2):
    '''Returns the average cross-spectrum between two spectra. Averages over the 2nd and 3rd
    dimension'''
    cross_spectrum = np.conj(complex_spectrum1) * complex_spectrum2
    return np.nanmean(cross_spectrum, axis=(1, 2)).squeeze()


@convert_pandas
def multitaper_power_spectral_density(data,
                                      sampling_frequency=1000,
                                      time_halfbandwidth_product=3,
                                      pad=0,
                                      tapers=None,
                                      frequencies=None,
                                      freq_ind=None,
                                      number_of_fft_samples=None,
                                      number_of_tapers=None,
                                      desired_frequencies=None):
    ''' Returns the multi-taper power spectral density of a time series
    '''
    tapers, number_of_fft_samples, frequencies, freq_ind = _set_default_multitaper_parameters(
        number_of_time_samples=data.shape[0],
        sampling_frequency=sampling_frequency,
        tapers=tapers,
        frequencies=frequencies,
        freq_ind=freq_ind,
        number_of_fft_samples=number_of_fft_samples,
        number_of_tapers=number_of_tapers,
        time_halfbandwidth_product=time_halfbandwidth_product,
        desired_frequencies=desired_frequencies,
        pad=pad)
    complex_spectrum = _multitaper_fft(
        tapers, _center_data(data), number_of_fft_samples, sampling_frequency)
    psd = np.real(_cross_spectrum(complex_spectrum[freq_ind, :, :],
                                  complex_spectrum[freq_ind, :, :])
                  )
    return pd.DataFrame({'power': psd,
                         'frequency': frequencies
                         }).set_index('frequency')


@convert_pandas
def multitaper_coherence(data, sampling_frequency=1000, time_halfbandwidth_product=3, pad=0,
                         number_of_tapers=None, desired_frequencies=None,
                         tapers=None, frequencies=None, freq_ind=None,
                         number_of_fft_samples=None):
    ''' Returns the multi-taper coherency of two time series
    data[0] (time x trials)
    data[1] (time x trials)
    '''
    tapers, number_of_fft_samples, frequencies, freq_ind = _set_default_multitaper_parameters(
        number_of_time_samples=data[0].shape[0],
        sampling_frequency=sampling_frequency,
        tapers=tapers,
        number_of_tapers=number_of_tapers,
        time_halfbandwidth_product=time_halfbandwidth_product,
        desired_frequencies=desired_frequencies,
        pad=pad)
    data = [_center_data(datum) for datum in data]
    complex_spectra = [_multitaper_fft(tapers, datum, number_of_fft_samples, sampling_frequency)
                       for datum in data]
    cross_spectrum = _cross_spectrum(complex_spectra[0][freq_ind, :, :],
                                     complex_spectra[1][freq_ind, :, :])
    spectrum = [_cross_spectrum(complex_spectrum[freq_ind, :, :],
                                complex_spectrum[freq_ind, :, :])
                for complex_spectrum in complex_spectra]

    coherency = cross_spectrum / np.sqrt(spectrum[0] * spectrum[1])
    return pd.DataFrame({'frequency': frequencies,
                         'coherence_magnitude': np.abs(coherency),
                         'coherence_phase': np.angle(coherency),
                         'power_spectrum1': np.real(spectrum[0]),
                         'power_spectrum2': np.real(spectrum[1])
                         }).set_index('frequency')


def _get_window_center(time_window_start, time_window_duration, time):
    return time[time_window_start] + (time_window_duration / 2)


@convert_pandas
def multitaper_coherogram(data,
                          sampling_frequency=1000,
                          time_window_duration=1,
                          time_window_step=None,
                          desired_frequencies=None,
                          time_halfbandwidth_product=3,
                          number_of_tapers=None,
                          pad=0,
                          tapers=None,
                          time=None):
    ''' Returns a pandas dataframe with the information for a coherogram.
    Sampling frequency and frequency resolution inputs are given in Hertz.
    Time window duration and steps are given in seconds.
    '''
    time_step_length, time_window_length = _get_window_lengths(
        time_window_duration,
        sampling_frequency,
        time_window_step)
    tapers, number_of_fft_samples, frequencies, freq_ind = _set_default_multitaper_parameters(
        number_of_time_samples=time_window_length,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        time_window_step=time_window_duration,
        tapers=tapers,
        number_of_tapers=number_of_tapers,
        time_halfbandwidth_product=time_halfbandwidth_product,
        desired_frequencies=desired_frequencies,
        pad=pad)
    return pd.concat(list(_make_sliding_window_dataframe(
        multitaper_coherence,
        data,
        time_window_duration,
        time_window_step,
        time_step_length,
        time_window_length,
        time,
        sampling_frequency=sampling_frequency,
        desired_frequencies=desired_frequencies,
        time_halfbandwidth_product=time_halfbandwidth_product,
        number_of_tapers=number_of_tapers,
        pad=pad,
        tapers=tapers,
        frequencies=frequencies,
        freq_ind=freq_ind,
        number_of_fft_samples=number_of_fft_samples,
    ))).sort_index()


def plot_coherogram(coherogram_dataframe, axis_handle=None,
                    cmap='viridis', vmin=0.3, vmax=0.7):
    if axis_handle is None:
        axis_handle = plt.gca()
    time, freq = _get_unique_time_freq(coherogram_dataframe)
    data2D = coherogram_dataframe.reset_index().pivot(
        'frequency', 'time', 'coherence_magnitude')
    mesh = axis_handle.pcolormesh(time, freq, data2D,
                                  cmap=cmap,
                                  shading='gouraud',
                                  vmin=vmin,
                                  vmax=vmax)
    axis_handle.set_xlim([time.min(), time.max()])
    axis_handle.set_ylim([freq.min(), freq.max()])
    return mesh


def plot_group_delayogram(coherogram_dataframe, axis_handle=None, cmap='RdBu',
                          vmin=-np.pi, vmax=np.pi, time_units='seconds', frequency_units='Hz'):
    if axis_handle is None:
        axis_handle = plt.gca()
    time, freq = _get_unique_time_freq(coherogram_dataframe)
    data2D = coherogram_dataframe.pivot('frequency', 'time', 'coherence_phase')
    mesh = axis_handle.pcolormesh(time, freq, data2D,
                                  cmap=cmap,
                                  shading='gouraud',
                                  vmin=vmin,
                                  vmax=vmax)
    axis_handle.set_ylabel('Frequency ({frequency_units})'.format(
        frequency_units=frequency_units))
    axis_handle.set_xlabel('Time ({time_units})'.format(time_units=time_units))
    axis_handle.set_xlim([time.min(), time.max()])
    axis_handle.set_ylim([freq.min(), freq.max()])
    return mesh


def coherence_title(tetrode_indices, cur_tetrode_info):
    return '{tetrode1} - {tetrode2}' \
        .format(tetrode1=tetrode_title(tetrode_indices[0], cur_tetrode_info),
                tetrode2=tetrode_title(tetrode_indices[1], cur_tetrode_info))


def group_delay(coherence_dataframe):
    slope, _, correlation, _, _ = scipy.stats.linregress(
        coherence_dataframe.reset_index('frequency').frequency,
        coherence_dataframe.coherence_phase)
    return pd.DataFrame({'correlation': correlation,
                         'number_of_points': len(coherence_dataframe.coherence_phase),
                         'slope': slope,
                         'delay': slope / 2 * np.pi}, index=[0])


def group_delay_over_time(coherogram_dataframe):
    return coherogram_dataframe.groupby('time').apply(group_delay)


def normalize_power_by_baseline(dataframe_of_interest, dataframe_baseline):
    '''Normalizes a coherence or power dataframe by dividing it by a baseline dataframe.
    The baseline dataframe is assumed to only have a frequency index.'''
    dropped_baseline = dataframe_baseline.drop(
        ['coherence_magnitude', 'coherence_phase'], axis=1, errors='ignore')
    dropped_doi = dataframe_of_interest.drop(
        ['coherence_magnitude', 'coherence_phase'], axis=1, errors='ignore')
    return dropped_doi / dropped_baseline


def normalize_coherence_by_baseline(dataframe_of_interest, dataframe_baseline):
    dropped_baseline = dataframe_baseline.drop(
        ['power_spectrum1', 'power_spectrum2'], axis=1, errors='ignore')
    dropped_doi = dataframe_of_interest.drop(
        ['power_spectrum1', 'power_spectrum2'], axis=1, errors='ignore')
    return dropped_doi - dropped_baseline


def difference_from_baseline_coherence(lfps, times_of_interest,
                                       baseline_window=(-0.250, 0),
                                       window_of_interest=(-0.250, .250),
                                       sampling_frequency=1000,
                                       time_window_duration=0.020,
                                       time_window_step=None,
                                       desired_frequencies=None,
                                       time_halfbandwidth_product=3,
                                       number_of_tapers=None):
    ''''''
    baseline_time_halfbandwidth_product = _match_frequency_resolution(
        time_halfbandwidth_product, time_window_duration, baseline_window)
    baseline_lfp_segments = list(
        ripple_detection.reshape_to_segments(
            lfps, times_of_interest, window_offset=baseline_window, concat_axis=1))
    time_of_interest_lfp_segments = list(
        ripple_detection.reshape_to_segments(
            lfps, times_of_interest, window_offset=window_of_interest, concat_axis=1))
    coherence_baseline = multitaper_coherence(
        baseline_lfp_segments,
        sampling_frequency=sampling_frequency,
        desired_frequencies=desired_frequencies,
        time_halfbandwidth_product=baseline_time_halfbandwidth_product)
    coherogram = multitaper_coherogram(
        time_of_interest_lfp_segments,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        time_window_step=time_window_step,
        desired_frequencies=desired_frequencies,
        time_halfbandwidth_product=time_halfbandwidth_product)
    return _normalize_power_coherence_by_baseline(coherogram, coherence_baseline)


def _match_frequency_resolution(time_halfbandwidth_product, time_window_duration, baseline_window):
    half_bandwidth = time_halfbandwidth_product / time_window_duration
    baseline_time_halfbandwidth_product = half_bandwidth * \
        (baseline_window[1] - baseline_window[0])
    if baseline_time_halfbandwidth_product < 1:
        raise AttributeError(
            'Baseline time-halfbandwidth product has to be greater than or equal to 1')
    else:
        return baseline_time_halfbandwidth_product


def _normalize_power_coherence_by_baseline(dataframe_of_interest, dataframe_baseline):
    normalized_power = normalize_power_by_baseline(dataframe_of_interest,
                                                   dataframe_baseline)
    normalized_coherence = normalize_coherence_by_baseline(dataframe_of_interest,
                                                           dataframe_baseline)
    return pd.concat([normalized_coherence, normalized_power], axis=1).sort_index()


def frequency_resolution(time_window_duration=None, time_halfbandwidth_product=None):
    return 2 * time_halfbandwidth_product / time_window_duration
