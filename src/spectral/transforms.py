import numpy as np
from scipy.fftpack import fft
from scipy.signal import detrend

from nitime.algorithms.spectral import dpss_windows


class Multitaper(object):

    def __init__(self, time_series, sampling_frequency=1000,
                 time_halfbandwidth_product=3, pad=0,
                 detrend_type='linear', time_window_duration=None,
                 time_window_step=None, n_tapers=None,  tapers=None,
                 start_time=0, n_fft_samples=None, n_time_samples=None,
                 n_samples_per_time_step=None):

        self.time_series = time_series
        self.sampling_frequency = sampling_frequency
        self.time_halfbandwidth_product = time_halfbandwidth_product
        self.pad = pad
        self.detrend_type = detrend_type
        self._time_window_duration = time_window_duration
        self._time_window_step = time_window_step
        self.start_time = start_time
        self._n_fft_samples = n_fft_samples
        self._tapers = tapers
        self._n_tapers = n_tapers
        self._n_time_samples = n_time_samples
        self._n_samples_per_time_step = n_samples_per_time_step

    @property
    def tapers(self):
        if self._tapers is None:
            self._tapers = _make_tapers(
                self.n_time_samples, self.sampling_frequency,
                self.time_halfbandwidth_product, self.n_tapers)
        return self._tapers

    @property
    def time_window_duration(self):
        if self._time_window_duration is None:
            self._time_window_duration = (self.n_time_samples /
                                          self.sampling_frequency)
        return self._time_window_duration

    @property
    def time_window_step(self):
        if self._time_window_step is None:
            self._time_window_step = (self.n_samples_per_time_step /
                                      self.sampling_frequency)
        return self._time_window_step

    @property
    def n_tapers(self):
        return np.floor(
            2 * self.time_halfbandwidth_product - 1).astype(int)

    @property
    def n_time_samples(self):
        if (self._n_time_samples is None and
                self._time_window_duration is None):
            self._n_time_samples = self.time_series.shape[0]
        elif self._time_window_duration is not None:
            self._n_time_samples = np.fix(
                self.time_window_duration * self.sampling_frequency
            ).astype(int)
        return self._n_time_samples

    @property
    def n_fft_samples(self):
        if self._n_fft_samples is None:
            next_exponent = _nextpower2(self.n_time_samples)
            self._n_fft_samples = max(2 ** (next_exponent + self.pad),
                                      self.n_time_samples)
        return self._n_fft_samples

    @property
    def frequencies(self):
        positive_frequencies = np.linspace(
            0, self.sampling_frequency, num=self.n_fft_samples // 2 + 1)
        return np.concatenate((positive_frequencies,
                               -1 * positive_frequencies[-2:0:-1]))

    @property
    def n_samples_per_time_step(self):
        '''If `time_window_step` is set, then calculate the
        `n_samples_per_time_step` based on the time window duration. If
        `time_window_step` and `n_samples_per_time_step` are both not set,
        default the window step size to the same size as the window.
        '''
        if (self._n_samples_per_time_step is None and
                self._time_window_step is None):
            self._n_samples_per_time_step = self.n_time_samples
        elif self._time_window_step is not None:
            self._n_samples_per_time_step = np.fix(
                self.time_window_step * self.sampling_frequency
            ).astype(int)
        return self._n_samples_per_time_step

    @property
    def time(self):
        time_ind = np.arange(
            0, self.time_series.shape[0] - 1,
            step=self.n_samples_per_time_step)
        return self.start_time + (time_ind / self.sampling_frequency)

    @property
    def n_signals(self):
        return self.time_series.shape[-1]

    @property
    def n_trials(self):
        return (1 if len(self.time_series.shape) < 3 else
                self.time_series.shape[1])

    @property
    def frequency_resolution(self):
        return (self.time_halfbandwidth_product /
                self.time_window_duration)

    def fft(self):
        time_series = _add_trial_axis(self.time_series)
        time_series = _sliding_window(
            time_series, window_size=self.n_time_samples,
            step_size=self.n_samples_per_time_step, axis=0)
        time_series = detrend(time_series, type=self.detrend_type)

        return _multitaper_fft(self.tapers, time_series,
                               self.n_fft_samples, self.sampling_frequency,
                               axis=3)


def _add_trial_axis(time_series):
    '''If no trial axis included, add one in
    '''
    return (time_series[:, np.newaxis, ...]
            if len(time_series.shape) < 3 else time_series)


def _sliding_window(data, window_size, step_size=1,
                    padded=False, axis=-1, is_copy=True):
    '''
    Calculate a sliding window over a signal

    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    window_size : int
        Number of samples per window
    step_size : int
        Number of samples to step the window forward. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    is_copy : bool
        Return strided array as copy to avoid sideffects when manipulating
        the output array.

    Returns
    -------
    data : array-like
        A matrix where row in last dimension consists of one instance
        of the sliding window.

    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.

    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> _sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> _sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])

    References
    ----------
    .. [1] https://gist.github.com/nils-werner/9d321441006b112a4b116a8387c2
    280c

    '''
    shape = list(data.shape)
    shape[axis] = np.floor(
        (data.shape[axis] / step_size) - (window_size / step_size) + 1
    ).astype(int)
    shape.append(window_size)

    strides = list(data.strides)
    strides[axis] *= step_size
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides)

    return strided.copy() if is_copy else strided


def _multitaper_fft(tapers, time_series, n_fft_samples,
                    sampling_frequency, axis=0):
    '''Projects the data on the tapers and returns the discrete Fourier
    transform

    Parameters
    ----------
    tapers : array_like, shape (n_time_samples, n_tapers)
    time_series : array_like, shape (n_windows, n_trials, n_time_samples)
    n_fft_samples : int
    sampling_frequency : int

    Returns
    -------
    fourier_coefficients : array_like, shape (n_windows, n_trials,
                                              n_fft_samples, n_tapers)

    '''
    projected_time_series = (
        np.reshape(time_series, (*time_series.shape, 1)) *
        np.reshape(tapers, (1, 1, *tapers.shape)))
    return (fft(projected_time_series, n=n_fft_samples, axis=axis) /
            sampling_frequency)


def _make_tapers(n_time_samples, sampling_frequency,
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


def _nextpower2(n):
    '''Return the next integer exponent of two greater than the given number.
    This is useful for ensuring fast FFT sizes.
    '''
    return np.ceil(np.log2(n)).astype(int)
