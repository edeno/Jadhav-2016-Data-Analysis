'''Functions for performing frequency domain analysis including:

- Multi-taper spectral analysis
- Coherence
- Canonical Coherence
- Group delay
- Parametric Spectral Granger

Multi-taper code is based on the Matlab library Chronux [1].

References
----------
.. [1] Bokil, H., Andrews, P., Kulkarni, J.E., Mehta, S., and Mitra, P.P.
   (2010). Chronux: a platform for analyzing neural signals.
   Journal of Neuroscience Methods 192, 146-151

'''
from functools import wraps
from inspect import signature
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.fftpack import fft
from scipy.ndimage import measurements
from scipy.stats import linregress, norm

from nitime.algorithms.spectral import dpss_windows


def convert_pandas(func):
    is_time = 'time' in signature(func).parameters.keys()

    @wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            if is_time:
                try:
                    kwargs['time'] = (data.index.get_level_values('time')
                                      .values)
                except AttributeError:
                    raise AttributeError(
                        'No time column or index provided in dataframe')
            data = data.values
        elif isinstance(data, list) and isinstance(data[0], pd.DataFrame):
            if is_time:
                try:
                    time = data[0].index.values
                    kwargs['time'] = time
                except AttributeError:
                    raise AttributeError(
                        'No time column or index provided in dataframe')
            data = [datum.values for datum in data]
        elif isinstance(data, list) and isinstance(data[0], pd.Panel):
            if is_time:
                try:
                    time = data[0].axes[1].values
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
            .format(
                tetrode_number=tetrode_index_tuple[-1],
                brain_area=cur_tetrode_info.loc[
                    tetrode_index_tuple, 'area']))


def _subtract_mean(x, axis=0):
    '''Returns the mean-centered data array along the first axis'''
    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning)
        return x - np.nanmean(x, axis=axis, keepdims=True)


def _get_window_array(data, time_window_start_ind, time_window_end_ind,
                      axis=0):
    '''Returns the data for a given start and end index'''
    slc = [slice(None)] * len(data[0].shape)
    slc[axis] = slice(time_window_start_ind, time_window_end_ind, 1)
    window_array = [datum[slc] for datum in data]
    if len(window_array) == 1:
        window_array = window_array[0]
    return window_array


def _make_sliding_window_dataframe(func, data, time_window_duration,
                                   time_window_step, time_step_length,
                                   time_window_length, time, axis,
                                   **kwargs):
    ''' Generator function that returns a transformed dataframe (via func)
    for each sliding time window.
    '''
    time_window_start_ind = 0
    while (time_window_start_ind +
           time_window_length) <= data[0].shape[axis]:
        try:
            time_window_end_ind = (time_window_start_ind +
                                   time_window_length)
            windowed_arrays = _get_window_array(
                data, time_window_start_ind, time_window_end_ind,
                axis=axis)

            yield (func(windowed_arrays, **kwargs)
                   .assign(
                time=_get_window_center(
                    time_window_start_ind, time_window_duration, time))
                   .set_index('time', append=True))
            time_window_start_ind += time_step_length
        except ValueError:
            # Not enough data points
            raise StopIteration


@convert_pandas
def multitaper_spectrogram(time_series, sampling_frequency=1000,
                           time_halfbandwidth_product=3,
                           time_window_duration=1, time_window_step=None,
                           pad=0, n_tapers=None, desired_frequencies=None,
                           tapers=None, time=None, n_fft_samples=None):
    '''Estimates the power spectral density of a time series using the
    multitaper method over time.

    The mean of each time series is automatically subtracted to remove the
    DC component.

    Parameters
    ----------
    time_series : array_like, shape=(n_time_samples, n_trials)
    sampling_frequency : int, optional
        Number of samples per second
    time_halfbandwidth_product : float, optional
        Controls the amount of smoothing in the time and frequency domains.
        It is equal to the duration of the time window multiplied by the
        desired half-bandwidth frequency resolution. If `n_tapers` is not
        set, then the number of tapers will be (2 *
        time_halfbandwidth_product) - 1.
    time_window_duration : float
        The duration of the sliding time window.
    time_window_step : float
        The amount the sliding time window advances.
    pad : int, optional
        Zero-pad the fft to the next power of two for computational
        efficiency. Setting this value to -1 results in no padding of the
        fft. The default (0) returns the Fourier frequencies.
    n_tapers : int, optional
        The number of tapers.
    desired_frequencies : array_like, optional
        A two-element array (low_frequency, high_frequency) that specifies
        the pass band of the returned power spectral density.

    Returns
    -------
    multitaper_spectrogram : Pandas Dataframe
        The spectral density as a function of frequency and time. Frequency
         and time are set as the index of the Dataframe.

    Other Parameters
    ----------------
    tapers : array_like, optional
        Allows the user to pass a pre-computed taper.
    time : array_like, optional
        The labels for the time axis.
    n_fft_samples : int, optional
        Allows the user to specify the number of fft samples.

    '''
    time_step_length, time_window_length = _get_window_lengths(
        time_window_duration,
        sampling_frequency,
        time_window_step)
    (tapers, n_fft_samples,
     frequencies, frequency_index) = _set_default_multitaper_parameters(
            n_time_samples=time_window_length,
            sampling_frequency=sampling_frequency,
            tapers=tapers,
            n_tapers=n_tapers,
            time_halfbandwidth_product=time_halfbandwidth_product,
            desired_frequencies=desired_frequencies,
            pad=pad)
    return pd.concat(list(_make_sliding_window_dataframe(
        multitaper_power_spectral_density,
        [time_series],
        time_window_duration,
        time_window_step,
        time_step_length,
        time_window_length,
        time,
        axis=0,
        sampling_frequency=sampling_frequency,
        desired_frequencies=desired_frequencies,
        time_halfbandwidth_product=time_halfbandwidth_product,
        n_tapers=n_tapers,
        pad=pad,
        tapers=tapers,
        frequencies=frequencies,
        frequency_index=frequency_index,
        n_fft_samples=n_fft_samples))
    ).sort_index()


def _set_default_multitaper_parameters(
    n_time_samples=None, sampling_frequency=None, tapers=None,
    n_tapers=None, time_halfbandwidth_product=None, pad=None,
    n_fft_samples=None, frequencies=None, frequency_index=None,
        desired_frequencies=None):
    '''Sets default multitaper parameters given that some
    subset of them are unset.

    Parameters
    ----------
    n_time_samples : int, optional
    sampling_frequency : int, optional
    tapers : array_like, optional
    n_tapers : int, optional
    time_halfbandwidth_product : float, optional
    pad : int, optional
    n_fft_samples : int, optional
    frequencies : array_like, optional
    frequency_index : array_like, optional
    desired_frequencies : array_like, optional

    Returns
    -------
    tapers : array_like
    n_fft_samples : int
    frequencies : array_like
    frequency_index : array_like

    '''
    if tapers is None:
        if n_tapers is None:
            n_tapers = int(np.floor(2 * time_halfbandwidth_product - 1))

        tapers = _get_tapers(n_time_samples, sampling_frequency,
                             time_halfbandwidth_product, n_tapers)
    if pad is None:
        pad = -1
    if n_fft_samples is None:
        next_exponent = _nextpower2(n_time_samples)
        n_fft_samples = max(2 ** (next_exponent + pad), n_time_samples)
    if frequencies is None:
        frequencies, frequency_index = _get_frequencies(
            sampling_frequency, n_fft_samples,
            desired_frequencies=desired_frequencies)
    return tapers, n_fft_samples, frequencies, frequency_index


def _get_window_lengths(time_window_duration, sampling_frequency,
                        time_window_step):
    '''Figures out the number of points per time window and step'''
    time_window_length = int(
        np.fix(time_window_duration * sampling_frequency))
    if time_window_step is None:
        time_window_step = time_window_duration
    time_step_length = int(np.fix(time_window_step * sampling_frequency))
    return time_step_length, time_window_length


def _get_unique_time_freq(spectrogram_dataframe):
    '''Returns the unique time and frequency given a spectrogram dataframe
    with non-unique time and frequency columns'''
    time = np.unique(spectrogram_dataframe.index.get_level_values('time'))
    half_time_diff = (time[1] - time[0]) / 2
    time = np.append(time - half_time_diff,
                     time[-1] + half_time_diff)
    frequency = np.unique(
        spectrogram_dataframe.index.get_level_values('frequency'))
    half_frequency_diff = (frequency[1] - frequency[0]) / 2
    frequency = np.append(frequency - half_frequency_diff,
                          frequency[-1] + half_frequency_diff)
    return time, frequency


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
    data2D = spectrogram_dataframe[spectrum_name].unstack()
    if plot_type is None:
        mesh = axis_handle.pcolormesh(time, freq, data2D, cmap=cmap,
                                      vmin=vmin, vmax=vmax)
    elif plot_type is 'change':
        mesh = axis_handle.pcolormesh(time, freq, data2D, cmap=cmap,
                                      norm=LogNorm(vmin=vmin, vmax=vmax))
    axis_handle.set_xlim([time.min(), time.max()])
    axis_handle.set_ylim([freq.min(), freq.max()])
    return mesh


def _get_frequencies(sampling_frequency, n_fft_samples,
                     desired_frequencies=None):
    ''' Returns the frequencies and frequency index. Desired frequencies is
    a two element tuple that defines the range of frequencies returned.
    If unset, all frequencies from 0 to the Nyquist
    (sampling_frequency / 2) are returned.
    '''
    if desired_frequencies is None:
        desired_frequencies = [0, sampling_frequency / 2]
    frequencies = np.linspace(0, sampling_frequency,
                              num=n_fft_samples + 1)
    frequency_ind = np.where((desired_frequencies[0] <= frequencies) &
                             (frequencies <= desired_frequencies[1]))[0]
    return frequencies[frequency_ind], frequency_ind


def _get_tapers(n_time_samples, sampling_frequency,
                time_halfbandwidth_product, n_tapers):
    '''Returns the Discrete prolate spheroidal sequences (tapers) for
    multi-taper spectral analysis.

    Parameters
    ----------
    n_time_samples : int
    sampling_frequency : int
    time_halfbandwidth_product : float
    n_tapers : int

    Returns
    -------
    tapers : array_like, shape (n_time_samples, n_tapers)

    '''
    tapers, _ = dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers)
    return tapers.T * np.sqrt(sampling_frequency)


def _multitaper_fft(tapers, time_series, n_fft_samples,
                    sampling_frequency):
    '''Projects the data on the tapers and returns the discrete Fourier
    transform

    Parameters
    ----------
    tapers : array_like, shape (n_time_samples, n_tapers)
    time_series : array_like, shape (n_time_samples, n_trials)
    n_fft_samples : int
    sampling_frequency : int

    Returns
    -------
    discrete_fourier_transform : array_like, shape (n_fft_samples,
                                                    n_trials, n_tapers)

    '''
    try:
        projected_time_series = (time_series[:, :, np.newaxis] *
                                 tapers[:, np.newaxis, :])
    except IndexError:  # There are no trials
        projected_time_series = (time_series[:, np.newaxis, np.newaxis] *
                                 tapers[:, :, np.newaxis])
    return (fft(projected_time_series, n=n_fft_samples, axis=0) /
            sampling_frequency)


def _nextpower2(n):
    '''Return the next integer exponent of two greater than the given number.
    This is useful for ensuring fast FFT sizes.
    '''
    return int(np.ceil(np.log2(n)))


def _cross_spectrum(complex_spectrum1, complex_spectrum2):
    '''Returns the average cross-spectrum between two spectra'''
    return np.conj(complex_spectrum1) * complex_spectrum2


@convert_pandas
def multitaper_power_spectral_density(
    time_series, sampling_frequency=1000, time_halfbandwidth_product=3,
    pad=0, tapers=None, frequencies=None, frequency_index=None,
        n_fft_samples=None, n_tapers=None, desired_frequencies=None):
    '''Estimates the power spectral density of a time series using the
    multitaper method.

    The time series is automatically centered.

    Parameters
    ----------
    time_series : array_like, shape=(n_time_samples, n_trials)
    sampling_frequency : int, optional
        Number of samples per second
    time_halfbandwidth_product : float, optional
        Controls the amount of smoothing in the time and frequency domains.
        It is equal to the duration of the time window multiplied by the
        desired half-bandwidth frequency resolution. If `n_tapers` is not
        set, then the number of tapers will be (2 *
        time_halfbandwidth_product) - 1.
    pad : int, optional
        Zero-pad the fft to the next power of two for computational
        efficiency. Setting this value to -1 results in no padding of the
        fft. The default (0) returns the Fourier frequencies.
    n_tapers : int, optional
        The number of tapers.
    desired_frequencies : array_like, optional
        A two-element array (low_frequency, high_frequency) that specifies
        the pass band of the returned power spectral density.

    Returns
    -------
    multitaper_power_spectral_density : Pandas Dataframe
        The spectral density as a function of frequency. Frequency is set
        as the index of the Dataframe.

    Other Parameters
    ----------------
    tapers : array_like, optional
        Allows the user to pass a pre-computed taper.
    frequencies : array_like, optional
        Allows the user to pass a pre-computed frequencies for the fft.
    frequency_index : array_like, optional
        Allows the user to pass a frequency index to limit the returned
        fft frequencies. This can be computed by setting the
        `desired_frequencies` parameter.
    n_fft_samples : int, optional
        Allows the user to specify the number of fft samples.

    '''
    tapers, n_fft_samples, frequencies, frequency_index = (
        _set_default_multitaper_parameters(
            n_time_samples=time_series.shape[0],
            sampling_frequency=sampling_frequency, tapers=tapers,
            frequencies=frequencies, frequency_index=frequency_index,
            n_fft_samples=n_fft_samples, n_tapers=n_tapers,
            time_halfbandwidth_product=time_halfbandwidth_product,
            desired_frequencies=desired_frequencies, pad=pad))
    complex_spectrum = _multitaper_fft(
        tapers, _subtract_mean(time_series), n_fft_samples,
        sampling_frequency)
    average_cross_spectrum = _average_over_trials_and_tapers(
        _cross_spectrum(complex_spectrum[frequency_index, :, :],
                        complex_spectrum[frequency_index, :, :]))
    return pd.DataFrame({'power': np.real(average_cross_spectrum),
                         'frequency': frequencies
                         }).set_index('frequency')


def _average_over_trials_and_tapers(time_series):
    '''Takes the average over trials (2nd dimension) and tapers
    (3rd dimension)
    '''
    return np.nanmean(time_series, axis=(1, 2)).squeeze()


@convert_pandas
def multitaper_coherence(time_series, sampling_frequency=1000,
                         time_halfbandwidth_product=3, pad=0,
                         n_tapers=None, desired_frequencies=None,
                         tapers=None, frequencies=None,
                         frequency_index=None, n_fft_samples=None):
    '''Estimates the frequency domain correlation of two time series

    The time series is automatically centered.

    Parameters
    ----------
    time_series : 2-element list of arrays of shape=(n_time_samples,
                                                     n_trials)
        Two time series of equal duration.
    sampling_frequency : int, optional
        Number of samples per second
    time_halfbandwidth_product : float, optional
        Controls the amount of smoothing in the time and frequency domains.
        It is equal to the duration of the time window multiplied by the
        desired half-bandwidth frequency resolution. If `n_tapers` is not
        set, then the number of tapers will be (2 *
        time_halfbandwidth_product) - 1.
    pad : int, optional
        Zero-pad the fft to the next power of two for computational
        efficiency. Setting this value to -1 results in no padding of the
        fft. The default (0) returns the Fourier frequencies.
    n_tapers : int, optional
        The number of tapers.
    desired_frequencies : array_like, optional
        A two-element array (low_frequency, high_frequency) that specifies
        the pass band of the returned coherence.

    Returns
    -------
    multitaper_coherence : Pandas DataFrame
        The DataFrame contains the magnitude of the coherence (not the
        magnitude-squared coherence), the phase of the coherence, and the
        power spectra of both signals. The index is frequency.

    Other Parameters
    ----------------
    tapers : array_like, optional
        Allows the user to pass a pre-computed taper.
    frequencies : array_like, optional
        Allows the user to pass a pre-computed frequencies for the fft.
    frequency_index : array_like, optional
        Allows the user to pass a frequency index to limit the returned
        fft frequencies. This can be computed by setting the
        `desired_frequencies` parameter.
    n_fft_samples : int, optional
        Allows the user to specify the number of fft samples.

    '''
    n_time_samples = time_series[0].shape[0]
    tapers, n_fft_samples, frequencies, frequency_index = (
        _set_default_multitaper_parameters(
            n_time_samples=n_time_samples,
            sampling_frequency=sampling_frequency, tapers=tapers,
            n_tapers=n_tapers,
            time_halfbandwidth_product=time_halfbandwidth_product,
            desired_frequencies=desired_frequencies, pad=pad))
    time_series = [_subtract_mean(series) for series in time_series]
    complex_spectra = [_multitaper_fft(
        tapers, series, n_fft_samples, sampling_frequency)
        for series in time_series]
    cross_spectrum = _average_over_trials_and_tapers(
        _cross_spectrum(complex_spectra[0][frequency_index, :, :],
                        complex_spectra[1][frequency_index, :, :]))
    spectrum = [_average_over_trials_and_tapers(
                    _cross_spectrum(
                        complex_spectrum[frequency_index, :, :],
                        complex_spectrum[frequency_index, :, :]))
                for complex_spectrum in complex_spectra]

    coherency = cross_spectrum / np.sqrt(spectrum[0] * spectrum[1])
    coherence_magnitude = np.abs(coherency)
    coherence_magnitude[
        np.where(coherence_magnitude >= 1)] = 1.0 - np.finfo(float).eps

    return pd.DataFrame({'frequency': frequencies,
                         'coherence_magnitude': coherence_magnitude,
                         'coherence_phase': np.angle(coherency),
                         'power_spectrum1': np.real(spectrum[0]),
                         'power_spectrum2': np.real(spectrum[1]),
                         'n_trials': _get_number_of_trials(time_series),
                         'n_tapers': tapers.shape[1],
                         'frequency_resolution': get_frequency_resolution(
                             (n_time_samples - 1) / sampling_frequency,
                             time_halfbandwidth_product)
                         }).set_index('frequency')


def _get_window_center(time_window_start, time_window_duration, time):
    return time[time_window_start] + (time_window_duration / 2)


def _get_number_of_trials(time_series):
    try:
        return time_series[0].shape[-1]
    except IndexError:
        return 1


@convert_pandas
def multitaper_coherogram(
    time_series, sampling_frequency=1000, time_window_duration=1,
    time_window_step=None, time_halfbandwidth_product=3, pad=0,
        n_tapers=None, desired_frequencies=None, tapers=None, time=None):
    '''Estimates the frequency domain correlation of two time series
    over sliding time windows.

    The time series are automatically centered for each time window

    Parameters
    ----------
    time_series : 2-element list of arrays of shape=(n_time_samples,
                                                     n_trials)
        Two time series of equal duration.
    sampling_frequency : int, optional
        Number of samples per second
    time_window_duration : float
        The duration of the sliding time window.
    time_window_step : float
        The amount the sliding time window advances.
    time_halfbandwidth_product : float, optional
        Controls the amount of smoothing in the time and frequency domains.
        It is equal to the duration of the time window multiplied by the
        desired half-bandwidth frequency resolution. If `n_tapers` is not
        set, then the number of tapers will be (2 *
        time_halfbandwidth_product) - 1.
    pad : int, optional
        Zero-pad the fft to the next power of two for computational
        efficiency. Setting this value to -1 results in no padding of the
        fft. The default (0) returns the Fourier frequencies.
    n_tapers : int, optional
        The number of tapers.
    desired_frequencies : array_like, optional
        A two-element array (low_frequency, high_frequency) that specifies
        the pass band of the returned coherence.

    Returns
    -------
    multitaper_coherogram : Pandas DataFrame
        The DataFrame contains the magnitude of the coherence (not the
        magnitude-squared coherence), the phase of the coherence, and the
        power spectra of both signals. The index is time and frequency.

    Other Parameters
    ----------------
    tapers : array_like, optional
        Allows the user to pass a pre-computed taper.
    time : array_like, optional
        The labels for the time axis.

    '''
    time_step_length, time_window_length = _get_window_lengths(
        time_window_duration,
        sampling_frequency,
        time_window_step)
    tapers, n_fft_samples, frequencies, frequency_index = (
        _set_default_multitaper_parameters(
            n_time_samples=time_window_length,
            sampling_frequency=sampling_frequency, tapers=tapers,
            n_tapers=n_tapers,
            time_halfbandwidth_product=time_halfbandwidth_product,
            desired_frequencies=desired_frequencies, pad=pad))
    return pd.concat(list(_make_sliding_window_dataframe(
        multitaper_coherence, time_series, time_window_duration,
        time_window_step, time_step_length, time_window_length, time,
        axis=0, sampling_frequency=sampling_frequency,
        desired_frequencies=desired_frequencies,
        time_halfbandwidth_product=time_halfbandwidth_product,
        n_tapers=n_tapers, pad=pad, tapers=tapers, frequencies=frequencies,
        frequency_index=frequency_index, n_fft_samples=n_fft_samples,
    ))).sort_index()


def plot_coherogram(coherogram_dataframe, axis_handle=None,
                    cmap='viridis', vmin=0, vmax=1):
    if axis_handle is None:
        axis_handle = plt.gca()
    time, freq = _get_unique_time_freq(coherogram_dataframe)
    data2D = coherogram_dataframe['coherence_magnitude'].unstack()
    mesh = axis_handle.pcolormesh(time, freq, data2D, cmap=cmap, vmin=vmin,
                                  vmax=vmax)
    axis_handle.set_xlim([time.min(), time.max()])
    axis_handle.set_ylim([freq.min(), freq.max()])
    return mesh


def plot_group_delayogram(coherogram_dataframe, axis_handle=None,
                          cmap='RdBu', vmin=-np.pi, vmax=np.pi):
    if axis_handle is None:
        axis_handle = plt.gca()
    time, freq = _get_unique_time_freq(coherogram_dataframe)
    data2D = coherogram_dataframe.reset_index().pivot(
        'frequency', 'time', 'coherence_phase')
    mesh = axis_handle.pcolormesh(time, freq, data2D,
                                  cmap=cmap,
                                  vmin=vmin,
                                  vmax=vmax)
    axis_handle.set_xlim([time.min(), time.max()])
    axis_handle.set_ylim([freq.min(), freq.max()])
    return mesh


def coherence_title(tetrode_indices, cur_tetrode_info):
    return '{tetrode1} - {tetrode2}'.format(
        tetrode1=tetrode_title(tetrode_indices[0], cur_tetrode_info),
        tetrode2=tetrode_title(tetrode_indices[1], cur_tetrode_info))


def group_delay(coherence_dataframe):
    coherence_dataframe = coherence_dataframe.dropna()
    frequency = coherence_dataframe.index.get_level_values('frequency')
    coherence_phase = np.unwrap(coherence_dataframe.coherence_phase)
    try:
        slope, _, correlation, _, _ = linregress(frequency,
                                                 coherence_phase)
    except ValueError:
        slope, correlation = np.nan, np.nan
    return pd.DataFrame({
        'correlation': correlation,
        'number_of_points': coherence_dataframe.shape[0],
        'slope': slope,
        'delay': slope / (2 * np.pi)
        }, index=[0])


def group_delay_over_time(coherogram_dataframe):
    return (coherogram_dataframe
            .groupby(level='time')
            .apply(group_delay)
            .reset_index(level=1, drop=True))


def power_change(baseline_power, power_of_interest):
    '''Normalizes a coherence or power dataframe by dividing it by a
    baseline dataframe. The baseline dataframe is assumed to only have a
    frequency index.'''
    baseline_power_dropped = baseline_power.drop(
        ['coherence_magnitude', 'coherence_phase', 'n_trials', 'n_tapers'],
        axis=1, errors='ignore')
    power_of_interest_dropped = power_of_interest.drop(
        ['coherence_magnitude', 'coherence_phase', 'n_trials', 'n_tapers'],
        axis=1, errors='ignore')
    return power_of_interest_dropped / baseline_power_dropped


def coherence_change(baseline_coherence, coherence_of_interest):
    coherence_of_interest_dropped = coherence_of_interest.drop(
        ['power_spectrum1', 'power_spectrum2', 'n_trials', 'n_tapers',
         'frequency_resolution'],
        axis=1, errors='ignore')
    baseline_coherence_dropped = baseline_coherence.drop(
        ['power_spectrum1', 'power_spectrum2', 'n_trials', 'n_tapers',
         'frequency_resolution'],
        axis=1, errors='ignore')
    return coherence_of_interest_dropped - baseline_coherence_dropped


@convert_pandas
def multitaper_canonical_coherence(
    time_series_groups, sampling_frequency=1000,
    time_halfbandwidth_product=3, pad=0, tapers=None, frequencies=None,
    frequency_index=None, n_fft_samples=None, n_tapers=None,
        desired_frequencies=None):
    '''Given two sets of signals, finds weights for each set of signals
    such that the linear combination of the signals is maximally coherent
    at a frequency.


    Parameters
    ----------
    time_series_groups : list of arrays, shape (n_signals, n_time_samples,
                                                n_trials)
        A two-element list that correspond to each set of signals
        respectively.
    sampling_frequency : int, optional
        Number of samples per second
    time_halfbandwidth_product : float, optional
        Controls the amount of smoothing in the time and frequency domains.
        It is equal to the duration of the time window multiplied by the
        desired half-bandwidth frequency resolution. If `n_tapers` is not
        set, then the number of tapers will be (2 *
        time_halfbandwidth_product) - 1.
    pad : int, optional
        Zero-pad the fft to the next power of two for computational
        efficiency. Setting this value to -1 results in no padding of the
        fft. The default (0) returns the Fourier frequencies.
    n_tapers : int, optional
        The number of tapers.
    desired_frequencies : array_like, optional
        A two-element array (low_frequency, high_frequency) that specifies
        the pass band of the returned coherence.

    Returns
    -------
    multitaper_canonical_coherence : Pandas DataFrame
        The DataFrame contains the magnitude of the coherence (not the
        magnitude-squared coherence), the phase of the coherence, and the
        power spectra of both signals. The index is frequency.

    Other Parameters
    ----------------
    tapers : array_like, optional
        Allows the user to pass a pre-computed taper.
    frequencies : array_like, optional
        Allows the user to pass a pre-computed frequencies for the fft.
    frequency_index : array_like, optional
        Allows the user to pass a frequency index to limit the returned
        fft frequencies. This can be computed by setting the
        `desired_frequencies` parameter.
    n_fft_samples : int, optional
        Allows the user to specify the number of fft samples.

    References
    ----------
    .. [1] Stephen, Emily Patricia. 2015. "Characterizing Dynamically
       Evolving Functional Networks in Humans with Application to Speech."
       Order No. 3733680, Boston University.
       http://search.proquest.com/docview/1731940762.

    '''

    group1_time_series, group2_time_series = (time_series_groups[0],
                                              time_series_groups[1])
    n_time_samples = group1_time_series[0].shape[0]
    tapers, n_fft_samples, frequencies, frequency_index = (
        _set_default_multitaper_parameters(
            n_time_samples=n_time_samples,
            sampling_frequency=sampling_frequency, tapers=tapers,
            n_tapers=n_tapers,
            time_halfbandwidth_product=time_halfbandwidth_product,
            desired_frequencies=desired_frequencies, pad=pad))
    complex_spectra1 = _get_complex_spectra(
        group1_time_series, tapers, n_fft_samples, sampling_frequency)
    complex_spectra2 = _get_complex_spectra(
        group2_time_series, tapers, n_fft_samples, sampling_frequency)
    canonical_coherency = np.asarray(
        [_estimate_canonical_coherency(complex_spectra1[:, frequency, :],
                                       complex_spectra2[:, frequency, :])
         for frequency in frequency_index], dtype=np.complex128)

    return pd.DataFrame(
        {'frequency': frequencies,
         'coherence_magnitude': np.abs(canonical_coherency),
         'coherence_phase': np.angle(canonical_coherency),
         'n_trials': _get_number_of_trials(time_series_groups),
         'n_tapers': tapers.shape[1],
         'frequency_resolution': get_frequency_resolution(
              (n_time_samples - 1) / sampling_frequency,
              time_halfbandwidth_product)
         }).set_index('frequency')


@convert_pandas
def multitaper_canonical_coherogram(time_series_groups,
                                    sampling_frequency=1000,
                                    time_window_duration=1,
                                    time_window_step=None,
                                    desired_frequencies=None,
                                    time_halfbandwidth_product=3,
                                    n_tapers=None,
                                    pad=0,
                                    tapers=None,
                                    time=None):
    time_step_length, time_window_length = _get_window_lengths(
        time_window_duration,
        sampling_frequency,
        time_window_step)
    tapers, n_fft_samples, frequencies, frequency_index = (
        _set_default_multitaper_parameters(
            n_time_samples=time_window_length,
            sampling_frequency=sampling_frequency, tapers=tapers,
            n_tapers=n_tapers,
            time_halfbandwidth_product=time_halfbandwidth_product,
            desired_frequencies=desired_frequencies, pad=pad))
    return pd.concat(list(
        _make_sliding_window_dataframe(
            multitaper_canonical_coherence, time_series_groups,
            time_window_duration, time_window_step, time_step_length,
            time_window_length, time, axis=1,
            sampling_frequency=sampling_frequency,
            desired_frequencies=desired_frequencies,
            time_halfbandwidth_product=time_halfbandwidth_product,
            n_tapers=n_tapers, pad=pad, tapers=tapers,
            frequencies=frequencies, frequency_index=frequency_index,
            n_fft_samples=n_fft_samples
        ))).sort_index()


def _get_complex_spectra(time_series, tapers, n_fft_samples,
                         sampling_frequency):
    '''Calculates the tapered Fourier coefficients for each signal.

    The time series are automatically centered by its mean over time.

    Parameters
    ----------
    time_series : array_like, shape (n_signals, n_time_samples, n_trials)
    tapers : array_like, shape (n_time_samples, n_tapers)
    n_fft_samples : int
    sampling_frequency : int

    Returns
    -------
    complex_spectra : array_like, shape (n_signals, n_fft_samples,
                                         n_trials * n_tapers)

    '''
    n_signals = time_series.shape[0]
    time_series = _subtract_mean(time_series, axis=1)
    complex_spectra = [_multitaper_fft(
        tapers, time_series[signal_ind, ...].squeeze(),
        n_fft_samples, sampling_frequency)
        for signal_ind in range(n_signals)]
    complex_spectra = np.concatenate(
        [spectra[np.newaxis, ...] for spectra in complex_spectra])
    return complex_spectra.reshape((n_signals, n_fft_samples, -1))


def _estimate_canonical_coherency(complex_spectra1, complex_spectra2):
    normalized_complex_spectra1 = _normalize_complex_spectra(
        complex_spectra1)
    normalized_complex_spectra2 = _normalize_complex_spectra(
        complex_spectra2)
    Q = _complex_dot_product(normalized_complex_spectra1,
                             normalized_complex_spectra2)
    return np.linalg.svd(Q, full_matrices=False, compute_uv=False)[0]


def _complex_dot_product(a, b):
    return np.dot(a, b.conj().transpose())


def _normalize_complex_spectra(complex_spectra):
    U1, _, V1 = np.linalg.svd(complex_spectra, full_matrices=False)
    return np.dot(U1, V1)


def _match_frequency_resolution(time_halfbandwidth_product,
                                time_window_duration, baseline_window):
    half_bandwidth = get_frequency_resolution(
        time_window_duration, time_halfbandwidth_product)
    baseline_time_halfbandwidth_product = (
        half_bandwidth * (baseline_window[1] - baseline_window[0]))
    if baseline_time_halfbandwidth_product < 1:
        raise AttributeError(
            'Baseline time-halfbandwidth product has to be greater than or'
            ' equal to 1')
    else:
        return baseline_time_halfbandwidth_product


def power_and_coherence_change(dataframe1, dataframe2):
    return (pd.concat([coherence_change(dataframe1, dataframe2),
                       power_change(dataframe1, dataframe2)], axis=1)
            .sort_index()
            .assign(n_trials_1=dataframe1.n_trials)
            .assign(n_trials_2=dataframe2.n_trials))


def get_frequency_resolution(time_window_duration,
                             time_halfbandwidth_product):
    '''Calculates the half-bandwith frequency resolution [-W, W].

    Given a frequency, other frequencies +/- the bandwidth are
    indistinguishable from each other due to the uncertainty principle.

    '''
    return time_halfbandwidth_product / time_window_duration


def _get_normal_distribution_p_values(data, mean=0, std_deviation=1):
        return 1 - norm.cdf(data, loc=mean, scale=std_deviation)


def _get_multitaper_bias(n_trials, n_tapers):
    '''The bias from performing `n_trials` * `n_tapers` estimates.

    In multitaper analysis, each trial and taper is an independent Fourier
    transform.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    n_tapers : int
        Number of tapers.

    Returns
    -------
    bias : int

    '''
    degrees_of_freedom = 2 * n_trials * n_tapers
    return 1 / (degrees_of_freedom - 2)


def fisher_z_transform(coherence_df):
    '''Transforms the coherence magnitude into an approximately
    standard normal test statistic, corrected for multitaper bias.
    '''
    bias = _get_multitaper_bias(coherence_df.n_trials,
                                coherence_df.n_tapers)
    test_statistic = (np.arctanh(coherence_df.coherence_magnitude) -
                      bias) / np.sqrt(bias)
    return pd.DataFrame(
        {'fisher_z': test_statistic,
         'p_value': _get_normal_distribution_p_values(test_statistic)
         })


def fisher_z_transform_difference(coherence_df1, coherence_df2):
    '''Compare the difference between coherence magnitudes.
    '''
    bias1 = _get_multitaper_bias(coherence_df1.n_trials,
                                 coherence_df1.n_tapers)
    bias2 = _get_multitaper_bias(coherence_df2.n_trials,
                                 coherence_df2.n_tapers)

    fisher_z_coherence1 = (np.arctanh(coherence_df1.coherence_magnitude) -
                           bias1)
    fisher_z_coherence2 = (np.arctanh(coherence_df2.coherence_magnitude) -
                           bias2)

    test_statistic = ((fisher_z_coherence2 - fisher_z_coherence1) /
                      np.sqrt(bias2 + bias1))

    return pd.DataFrame(
        {'fisher_z': test_statistic,
         'p_value': _get_normal_distribution_p_values(test_statistic)
         })


def filter_significant_groups_less_than_frequency_resolution(
        is_significant, frequency_resolution):
    '''Finds clusters of statistical significance and ensures that they
    are greater than the frequency resolution.

    Also makes sure significant frequency points are independent.

    This is important for calculating group delay because accurate
    calculation requires the phase of multiple frequencies and frequencies
    within the frequency resolution are indistinguishable.

    This function works by labeling the clusters of significant tests --
    adjacent True values bordered by False values -- and determining their
    size. If their size is greater than the frequency resolution, then
    the group is left unchanged. If the size is less than the frequency
    resolution, the group is changed to False values.

    Parameters
    ----------
    is_significant : boolean Pandas series
        A Pandas data series that is True for passing tests and False for
        tests that failed to pass. Frequency must be a level of the index.
    frequency_resolution : float
        The half-bandwith frequency resolution

    Returns
    -------
    is_significant_corrected : boolean Pandas series
        A Pandas series the same size as `is_significant` with clusters of
        significance that are smaller than the frequency resolution set to
        False.

    Examples
    --------
    z_coherence = fisher_z_transform(coherogram)
    frequency_resolution = coherogram.frequency_resolution.unique()[0]
    is_significant = (pd.Series(
        adjust_for_multiple_comparisons(z_coherence.p_value),
        index=z_coherence.index)
        .groupby(level='time')
        .transform(
            get_significant_groups_greater_than_frequency_resolution,
            frequency_resolution))
    group_delay_over_time(coherogram.mask(~is_significant))

    '''
    frequencies = is_significant.index.get_level_values('frequency')
    frequency_change = frequencies[1] - frequencies[0]
    significant_groups, _ = measurements.label(is_significant)
    independent_frequency_points = np.ceil(
        frequency_resolution / frequency_change).astype(int)

    def _less_than_frequency_resolution(significant_group):
        n_significant_points = significant_group.shape[0]
        position_index = np.arange(0, n_significant_points)
        if ((n_significant_points - 1) * frequency_change <=
                frequency_resolution):
            significant_group.iloc[position_index] = False
        else:
            good_index = np.arange(0, n_significant_points,
                                   independent_frequency_points)
            bad_ind = np.setdiff1d(position_index, good_index)
            significant_group.iloc[bad_ind] = False
        return significant_group

    return (is_significant
            .groupby(significant_groups)
            .transform(_less_than_frequency_resolution)
            .sort_index())
