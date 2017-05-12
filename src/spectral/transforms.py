import numpy as np
from scipy import interpolate
from scipy.fftpack import fft, ifft
from scipy.linalg import eigvals_banded
from scipy.signal import detrend


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


def tridisolve(d, e, b, overwrite_b=True):
    '''Symmetric tridiagonal system solver, from Golub and Van Loan p157.
    .. note:: Copied from NiTime.
    Parameters
    ----------
    d : ndarray
      main diagonal stored in d[:]
    e : ndarray
      superdiagonal stored in e[:-1]
    b : ndarray
      RHS vector
    Returns
    -------
    x : ndarray
      Solution to Ax = b (if overwrite_b is False). Otherwise solution is
      stored in previous RHS vector b
    '''
    N = len(b)
    # work vectors
    dw = d.copy()
    ew = e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    for k in range(1, N):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    for k in range(1, N):
        x[k] = x[k] - ew[k - 1] * x[k - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for k in range(N - 2, -1, -1):
        x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    if not overwrite_b:
        return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    '''Perform an inverse iteration.
    This will find the eigenvector corresponding to the given eigenvalue
    in a symmetric tridiagonal system.
    ..note:: Copied from NiTime.
    Parameters
    ----------
    d : ndarray
      main diagonal of the tridiagonal system
    e : ndarray
      offdiagonal stored in e[:-1]
    w : float
      eigenvalue of the eigenvector
    x0 : ndarray
      initial point to start the iteration
    rtol : float
      tolerance for the norm of the difference of iterates
    Returns
    -------
    e: ndarray
      The converged eigenvector
    '''
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0


def dpss_windows(n_time_samples, time_halfbandwidth_product, n_tapers,
                 low_bias=True, interp_from=None, interp_kind='linear'):
    '''Compute Discrete Prolate Spheroidal Sequences.

    Will give of orders [0, n_tapers-1] for a given frequency-spacing
    multiple NW and sequence length `n_time_samples`.

    Copied from NiTime and MNE-Python

    Parameters
    ----------
    n_time_samples : int
        Sequence length
    time_halfbandwidth_product : float, unitless
        Standardized half bandwidth corresponding to 2 * half_bw = BW * f0
        = BW * `n_time_samples` / dt but with dt taken as 1
    n_tapers : int
        Number of DPSS windows to return
    low_bias : Bool
        Keep only tapers with eigenvalues > 0.9
    interp_from : int (optional)
        The dpss can be calculated using interpolation from a set of dpss
        with the same NW and n_tapers, but shorter n_time_samples.
        This is the length of the shorter set of dpss windows.
    interp_kind : str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear',
        'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an integer
        specifying the order of the spline interpolator to use.

    Returns
    -------
    tapers, eigenvalues : tuple,
        v is an array of DPSS windows shaped (n_tapers, n_time_samples)
        e are the eigenvalues

    Notes
    -----
    Tridiagonal form of DPSS calculation from:
    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430

    '''
    n_tapers = int(n_tapers)
    W = float(time_halfbandwidth_product) / n_time_samples
    nidx = np.arange(n_time_samples, dtype='d')

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size
    # (n_time_samples)
    if interp_from is not None:
        if interp_from > n_time_samples:
            e_s = 'In dpss_windows, interp_from is: %s ' % interp_from
            e_s += 'and n_time_samples is: %s. ' % n_time_samples
            e_s += 'Please enter interp_from smaller than n_time_samples.'
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, time_halfbandwidth_product,
                            n_tapers, low_bias=False)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            I = interpolate.interp1d(x, this_d, kind=interp_kind)
            d_temp = I(np.linspace(
                0, this_d.shape[-1] - 1, n_time_samples, endpoint=False))

            # Rescale:
            d_temp = d_temp / np.sqrt(sum_squared(d_temp))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        '''here we want to set up an optimization problem to find a sequence
        whose energy is maximally concentrated within band [-W,W]. Thus,
        the measure lambda(T,W) is the ratio between the energy within
        that band, and the total energy. This leads to the eigen-system
        (A - (l1)I)v = 0, where the eigenvector corresponding to the
        largest eigenvalue is the sequence with maximally concentrated
        energy. The collection of eigenvectors of this system are called
        Slepian sequences, or discrete prolate spheroidal sequences (DPSS).
        Only the first K, K = 2NW/dt orders of DPSS will exhibit good
        spectral concentration
        [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

        Here I set up an alternative symmetric tri-diagonal eigenvalue
        problem such that
        (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
        the main diagonal = ([n_time_samples-1-2*t]/2)**2 cos(2PIW),
        t=[0,1,2,...,n_time_samples-1] and the first off-diagonal =
        t(n_time_samples-t)/2, t=[1,2,...,n_time_samples-1]
        [see Percival and Walden, 1993]'''
        diagonal = ((n_time_samples - 1 - 2 * nidx) / 2.) ** 2 * np.cos(
            2 * np.pi * W)
        off_diag = np.zeros_like(nidx)
        off_diag[:-1] = nidx[1:] * (n_time_samples - nidx[1:]) / 2.
        # put the diagonals in LAPACK 'packed' storage
        ab = np.zeros((2, n_time_samples), 'd')
        ab[1] = diagonal
        ab[0, 1:] = off_diag[:-1]
        # only calculate the highest n_tapers eigenvalues
        w = eigvals_banded(
            ab, select='i', select_range=(n_time_samples - n_tapers,
                                          n_time_samples - 1))
        w = w[::-1]

        # find the corresponding eigenvectors via inverse iteration
        t = np.linspace(0, np.pi, n_time_samples)
        dpss = np.zeros((n_tapers, n_time_samples), 'd')
        for k in range(n_tapers):
            dpss[k] = tridi_inverse_iteration(diagonal, off_diag, w[k],
                                              x0=np.sin((k + 1) * t))

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2 * i] *= -1
    # rather than test the sign of one point, test the sign of the
    # linear slope up to the first (largest) peak
    pk = np.argmax(np.abs(dpss[1::2, :n_time_samples // 2]), axis=1)
    for i, p in enumerate(pk):
        if np.sum(dpss[2 * i + 1, :p]) < 0:
            dpss[2 * i + 1] *= -1

    '''Now find the eigenvalues of the original spectral concentration
    problem Use the autocorr sequence technique from Percival and Walden,
    1993 pg 390'''

    # compute autocorr using FFT (same as nitime.utils.autocorr(dpss) *
    # n_time_samples)
    rxx_size = 2 * n_time_samples - 1
    n_fft = 2 ** int(np.ceil(np.log2(rxx_size)))
    dpss_fft = fft(dpss, n_fft)
    dpss_rxx = np.real(ifft(dpss_fft * dpss_fft.conj()))
    dpss_rxx = dpss_rxx[:, :n_time_samples]

    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    eigvals = np.dot(dpss_rxx, r)

    if low_bias:
        idx = (eigvals > 0.9)
        if not idx.any():
            print('Could not properly use low_bias,'
                  'keeping lowest-bias taper')
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    assert dpss.shape[1] == n_time_samples  # old nitime bug
    return dpss, eigvals


def sum_squared(X):
    '''Compute norm of an array.

    Parameters
    ----------
    X : array
        Data whose norm must be found

    Returns
    -------
    value : float
        Sum of squares of the input array X
    '''
    X_flat = X.ravel(order='F' if np.isfortran(X) else 'C')
    return np.dot(X_flat, X_flat)
