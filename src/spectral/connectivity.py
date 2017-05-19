from functools import partial
from inspect import signature
from itertools import combinations

import numpy as np
from scipy.ndimage import label
from scipy.stats.mstats import linregress

from .minimum_phase_decomposition import minimum_phase_decomposition
from .statistics import (adjust_for_multiple_comparisons,
                         fisher_z_transform,
                         get_normal_distribution_p_values)

EXPECTATION = {
    'trials': partial(np.mean, axis=1),
    'tapers': partial(np.mean, axis=2),
    'trials_tapers': partial(np.mean, axis=(1, 2))
}


class lazyproperty:

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class Connectivity(object):
    '''

    Attributes
    ----------
    fourier_coefficients : array
        The compex-valued coefficients from a fourier transform.
    expectation_type : ('trials_tapers' | 'trials' | 'tapers')
        How to average the cross spectral matrix.
    frequencies_of_interest : array
    '''

    def __init__(self, fourier_coefficients, frequencies_of_interest=None,
                 expectation_type='trials_tapers'):
        self.fourier_coefficients = fourier_coefficients
        self.frequencies_of_interest = frequencies_of_interest
        self.expectation_type = expectation_type

    @lazyproperty
    def cross_spectral_matrix(self):
        '''

        Parameters
        ----------
        fourier_coefficients : array, shape (n_time_samples, n_trials,
                                             n_signals, n_fft_samples,
                                             n_tapers)

        Returns
        -------
        cross_spectral_matrix : array, shape (..., n_signals, n_signals)

        '''
        fourier_coefficients = (self.fourier_coefficients
                                .swapaxes(2, -1)[..., np.newaxis])
        return _complex_inner_product(fourier_coefficients,
                                      fourier_coefficients)

    @lazyproperty
    def power(self):
        fourier_coefficients = self.fourier_coefficients.swapaxes(2, -1)
        return self.expectation(fourier_coefficients *
                                fourier_coefficients.conjugate()).real

    @lazyproperty
    def minimum_phase_factor(self):
        return minimum_phase_decomposition(
            self.expectation(self.cross_spectral_matrix))

    @lazyproperty
    def transfer_function(self):
        return _estimate_transfer_function(self.minimum_phase_factor)

    @lazyproperty
    def noise_covariance(self):
        return _estimate_noise_covariance(self.minimum_phase_factor)

    @lazyproperty
    def MVAR_Fourier_coefficients(self):
        return np.linalg.inv(self.transfer_function)

    @property
    def expectation(self):
        return EXPECTATION[self.expectation_type]

    @property
    def n_observations(self):
        axes = signature(self.expectation).parameters['axis'].default
        if isinstance(axes, int):
            return self.cross_spectral_matrix.shape[axes]
        else:
            return np.prod(
                [self.cross_spectral_matrix.shape[axis]
                 for axis in axes])

    @property
    def bias(self):
        degrees_of_freedom = 2 * self.n_observations
        return 1 / (degrees_of_freedom - 2)

    def coherency(self):
        '''The complex-valued linear association between time series in the
         frequency domain

         Returns
         -------
         complex_coherency : array, shape (..., n_fft_samples, n_signals,
                                           n_signals)

         '''
        return self.expectation(self.cross_spectral_matrix) / np.sqrt(
            self.power[..., :, np.newaxis] *
            self.power[..., np.newaxis, :])

    def coherence_phase(self):
        '''The phase angle of the complex coherency

        Returns
        -------
        phase : array, shape (..., n_fft_samples, n_signals, n_signals)

        '''
        return np.angle(self.coherency())

    def coherence_magnitude(self):
        '''The magnitude of the complex coherency.

        Note that this is not the magnitude squared coherence.

        Returns
        -------
        magnitude : array, shape (..., n_fft_samples, n_signals, n_signals)

        '''
        return _squared_magnitude(self.coherency())

    def imaginary_coherence(self):
        '''The normalized imaginary component of the cross-spectrum.

        Projects the cross-spectrum onto the imaginary axis to mitigate the
        effect of volume-conducted dependencies. Assumes volume-conducted
        sources arrive at sensors at the same time, resulting in
        a cross-spectrum with phase angle of 0 (perfectly in-phase) or \pi
        (anti-phase) if the sensors are on opposite sides of a dipole
        source. With the imaginary coherence, in-phase and anti-phase
        associations are set to zero.

        Returns
        -------
        imaginary_coherence_magnitude : array, shape (..., n_fft_samples,
                                                      n_signals, n_signals)

        References
        ----------
        .. [1] Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., and
               Hallett, M. (2004). Identifying true brain interaction from
               EEG data using the imaginary part of coherency. Clinical
               Neurophysiology 115, 2292-2307.

        '''
        return np.abs(
            self.expectation(self.cross_spectral_matrix).imag /
            np.sqrt(self.power[..., :, np.newaxis] *
                    self.power[..., np.newaxis, :]))

    def canonical_coherence(self, group_labels):
        '''Finds the maximal coherence between all combinations of groups.

        The canonical coherence finds two sets of weights such that the
        coherence between the linear combination of group1 and the linear
        combination of group2 is maximized.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Links each signal to a group.

        Returns
        -------
        canonical_coherence : array, shape (..., n_fft_samples,
                                            n_group_pairs)
            The maximimal coherence for each group pair where
            n_group_pairs = (n_groups) * (n_groups - 1) / 2
        pair_labels : list of tuples, shape (n_group_pairs, 2)
            The label for the group pair for which the coherence was
            maximized.

        References
        ----------
        .. [1] Stephen, E.P. (2015). Characterizing dynamically evolving
               functional networks in humans with application to speech.
               Boston University.

        '''
        labels = np.unique(group_labels)
        normalized_fourier_coefficients = [
            _normalize_fourier_coefficients(
                self.fourier_coefficients[
                    :, :, np.in1d(group_labels, label), ...])
            for label in labels]
        coherence = _squared_magnitude(np.stack([
            _estimate_canonical_coherency(
                fourier_coefficients1, fourier_coefficients2)
            for fourier_coefficients1, fourier_coefficients2
            in combinations(normalized_fourier_coefficients, 2)
        ], axis=-1)
        pair_labels = list(combinations(labels, 2))
        return coherence, pair_labels

    def phase_locking_value(self):
        '''The cross-spectrum with the power for each signal scaled to
        a magnitude of 1.

        The phase locking value attempts to mitigate power differences
        between realizations (tapers or trials) by treating all values of
        the cross-spectrum as the same power. This has the effect of
        downweighting high power realizations and upweighting low power
        realizations.

        Returns
        -------
        phase_locking_value : array, shape (..., n_fft_samples, n_signals,
                                            n_signals)

        References
        ----------
        .. [1] Lachaux, J.-P., Rodriguez, E., Martinerie, J., Varela, F.J.,
               and others (1999). Measuring phase synchrony in brain
               signals. Human Brain Mapping 8, 194-208.

        '''
        return self.expectation(
            self.cross_spectral_matrix /
            np.abs(self.cross_spectral_matrix))

    def phase_lag_index(self):
        '''A non-parametric synchrony measure designed to mitigate power
        differences between realizations (tapers, trials) and
        volume-conduction.

        The phase lag index is the average sign of the imaginary
        component of the cross-spectrum. The imaginary component sets
        in-phase or anti-phase signals to zero and the sign scales it to
        have the same magnitude regardless of phase.

        Returns
        -------
        phase_lag_index : array, shape (..., n_fft_samples, n_signals,
                                        n_signals)

        References
        ----------
        .. [1] Stam, C.J., Nolte, G., and Daffertshofer, A. (2007). Phase
               lag index: Assessment of functional connectivity from multi
               channel EEG and MEG with diminished bias from common
               sources. Human Brain Mapping 28, 1178-1193.

        '''
        return self.expectation(
            np.sign(self.cross_spectral_matrix.imag))

    def weighted_phase_lag_index(self):
        '''Weighted average of the phase lag index using the imaginary
        coherency magnitudes as weights.

        Returns
        -------
        weighted_phase_lag_index : array, shape (..., n_fft_samples,
                                                 n_signals, n_signals)

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        '''
        pli = self.phase_lag_index()
        weights = self.expectation(
            np.abs(self.cross_spectral_matrix.imag))
        with np.errstate(divide='ignore', invalid='ignore'):
            return pli / weights

    def debiased_squared_phase_lag_index(self):
        '''The square of the phase lag index corrected for the positive
        bias induced by using the magnitude of the complex cross-spectrum.

        Returns
        -------
        phase_lag_index : array, shape (..., n_fft_samples, n_signals,
                                        n_signals)

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        '''
        n_observations = self.n_observations
        return ((n_observations * self.phase_lag_index() ** 2 - 1.0) /
                (n_observations - 1.0))

    def debiased_squared_weighted_phase_lag_index(self):
        '''The square of the weighted phase lag index corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        weighted_phase_lag_index : array, shape (..., n_fft_samples,
                                                 n_signals, n_signals)

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        '''
        n_observations = self.n_observations
        imaginary_cross_spectral_matrix_sum = self.expectation(
            self.cross_spectral_matrix.imag) * n_observations
        squared_imaginary_cross_spectral_matrix_sum = self.expectation(
            self.cross_spectral_matrix.imag ** 2) * n_observations
        imaginary_cross_spectral_matrix_magnitude_sum = self.expectation(
            np.abs(self.cross_spectral_matrix.imag)) * n_observations
        weights = (imaginary_cross_spectral_matrix_magnitude_sum ** 2 -
                   squared_imaginary_cross_spectral_matrix_sum)
        return (imaginary_cross_spectral_matrix_sum ** 2 -
                squared_imaginary_cross_spectral_matrix_sum) / weights

    def pairwise_phase_consistency(self):
        '''The square of the phase locking value corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        phase_locking_value : array, shape (..., n_fft_samples, n_signals,
                                            n_signals)

        References
        ----------
        .. [1] Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., and
               Pennartz, C.M.A. (2010). The pairwise phase consistency: A
               bias-free measure of rhythmic neuronal synchronization.
               NeuroImage 51, 112-122.

        '''
        n_observations = self.n_observations
        plv_sum = self.phase_locking_value() * n_observations
        ppc = ((plv_sum * plv_sum.conjugate() - n_observations) /
               (n_observations * (n_observations - 1.0)))
        return ppc.real

    def spectral_granger_prediction(self):
        '''The amount of power at a node in a frequency explained by (is
        predictive of) the power at other nodes.

        References
        ----------
        .. [1] Geweke, J. (1982). Measurement of Linear Dependence and
               Feedback Between Multiple Time Series. Journal of the
               American Statistical Association 77, 304.

        '''
        partial_covariance = _remove_instantaneous_causality(
            self.noise_covariance)
        intrinsic_power = (self.power[..., np.newaxis] -
                           partial_covariance *
                           _squared_magnitude(self.transfer_function))
        return np.log(self.power[..., np.newaxis] / intrinsic_power)

    def directed_transfer_function(self):
        '''The transfer function coupling strength normalized by the total
        influence of other signals on that signal (inflow).

        Characterizes the direct and indirect coupling to a node.

        Returns
        -------
        directed_transfer_function : array, shape (..., n_fft_samples,
                                                   n_signals, n_signals)

        References
        ----------
        .. [1] Kaminski, M., and Blinowska, K.J. (1991). A new method of
               the description of the information flow in the brain
               structures. Biological Cybernetics 65, 203-210.

        '''

        return (_squared_magnitude(self.transfer_function) /
                _total_inflow(self.transfer_function))

    def directed_coherence(self):
        '''The transfer function coupling strength normalized by the total
        influence of other signals on that signal (inflow).

        This measure is the same as the directed transfer function but the
        signal inflow is scaled by the noise variance.

        Returns
        -------
        directed_coherence : array, shape (..., n_fft_samples,
                                           n_signals, n_signals)

        References
        ----------
        .. [1] Baccala, L., Sameshima, K., Ballester, G., Do Valle, A., and
               Timo-Iaria, C. (1998). Studying the interaction between
               brain structures via directed coherence and Granger
               causality. Applied Signal Processing 5, 40.

        '''
        noise_variance = _get_noise_variance(self.noise_covariance)
        return (np.sqrt(noise_variance) *
                _squared_magnitude(self.transfer_function) /
                _total_inflow(self.transfer_function, noise_variance))

    def partial_directed_coherence(self):
        '''The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        Returns
        -------
        partial_directed_coherence : array, shape (..., n_fft_samples,
                                                   n_signals, n_signals)

        References
        ----------
        .. [1] Baccala, L.A., and Sameshima, K. (2001). Partial directed
               coherence: a new concept in neural structure determination.
               Biological Cybernetics 84, 463-474.

        '''
        return (_squared_magnitude(self.MVAR_Fourier_coefficients) /
                _total_outflow(self.MVAR_Fourier_coefficients, 1.0))

    def generalized_partial_directed_coherence(self):
        '''The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        The generalized partial directed coherence scales the relative
        strength of coupling by the noise variance.

        Returns
        -------
        generalized_partial_directed_coherence : array,
                                                 shape (..., n_fft_samples,
                                                        n_signals,
                                                        n_signals)

        References
        ----------
        .. [1] Baccala, L.A., Sameshima, K., and Takahashi, D.Y. (2007).
               Generalized partial directed coherence. In Digital Signal
               Processing, 2007 15th International Conference on, (IEEE),
               pp. 163-166.

        '''
        noise_variance = _get_noise_variance(self.noise_covariance)
        return (_squared_magnitude(self.MVAR_Fourier_coefficients) /
                np.sqrt(noise_variance) / _total_outflow(
                    self.MVAR_Fourier_coefficients, noise_variance))

    def direct_directed_transfer_function(self):
        '''A combination of the directed transfer function estimate of
        directional influence between signals and the partial coherence's
        accounting for the influence of other signals.

        Returns
        -------
        direct_directed_transfer_function : array, shape
                                            (..., n_fft_samples,
                                             n_signals, n_signals)

        References
        ----------
        .. [1] Korzeniewska, A., Manczak, M., Kaminski,
               M., Blinowska, K.J., and Kasicki, S. (2003). Determination
               of information flow direction among brain structures by a
               modified directed transfer function (dDTF) method.
               Journal of Neuroscience Methods 125, 195-207.

        '''
        full_frequency_DTF = (_squared_magnitude(self.transfer_function) /
                              np.sum(_total_inflow(self.transfer_function),
                              axis=-3, keepdims=True))
        return full_frequency_DTF * self.partial_directed_coherence()

    def group_delay(self, frequencies_of_interest=None,
                    frequencies=None, frequency_resolution=None):
        '''The average time-delay of a broadband signal.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,)
        frequencies : array-like, shape (n_fft_samples,)
        frequency_resolution : float

        Returns
        -------
        delay : array, shape (..., n_signals, n_signals)
        slope : array, shape (..., n_signals, n_signals)
        r_value : array, shape (..., n_signals, n_signals)

        References
        ----------
        .. [1] Gotman, J. (1983). Measurement of small time differences
               between EEG channels: method and application to epileptic
               seizure propagation. Electroencephalography and Clinical
               Neurophysiology 56, 501-514.

        '''
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution)
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest)
        is_significant = _find_significant_frequencies(
            bandpassed_coherency, self.bias, independent_frequency_step)
        coherence_phase = np.ma.masked_array(
            np.unwrap(np.angle(bandpassed_coherency), axis=-3),
            mask=~is_significant)

        def _linear_regression(response):
            return linregress(bandpassed_frequencies, y=response)

        regression_results = np.ma.apply_along_axis(
            _linear_regression, -3, coherence_phase)
        slope = np.array(regression_results[..., 0, :, :], dtype=np.float)
        delay = slope / (2 * np.pi)
        r_value = np.array(
            regression_results[..., 2, :, :], dtype=np.float)
        return delay, slope, r_value

    def phase_slope_index(self, frequencies_of_interest=None,
                          frequencies=None, frequency_resolution=None):
        '''The weighted average of slopes of a broadband signal projected
        onto the imaginary axis.

        The phase slope index finds the complex weighted average of the
        coherency between frequencies where the weights correspond to the
        magnitude of the coherency at that frequency. This is projected
        on to the imaginary axis to avoid volume conduction effects.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,)
        frequencies : array-like, shape (n_fft_samples,)
        frequency_resolution : float

        Returns
        -------
        phase_slope_index : array, shape (..., n_signals, n_signals)

        References
        ----------
        .. [1] Nolte, G., Ziehe, A., Nikulin, V.V., Schlogl, A., Kramer,
               N., Brismar, T., and Muller, K.-R. (2008). Robustly
               Estimating the Flow Direction of Information in Complex
               Physical Systems. Physical Review Letters 100.

        '''
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest)

        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution)
        frequency_index = np.arange(0, bandpassed_frequencies.shape[0],
                                    independent_frequency_step)
        bandpassed_coherency = bandpassed_coherency[
            ..., frequency_index, :, :]

        return _inner_combination(bandpassed_coherency).imag


def _inner_combination(data, axis=-3):
    '''Takes the inner product of all possible pairs of a
    dimension without regard to order (combinations)'''
    combination_index = np.array(
        list(combinations(range(data.shape[axis]), 2)))
    combination_slice1 = np.take(data, combination_index[:, 0], axis)
    combination_slice2 = np.take(data, combination_index[:, 1], axis)
    return (combination_slice1.conjugate() * combination_slice2).sum(
        axis=axis)


def _estimate_noise_covariance(minimum_phase):
    A_0 = minimum_phase[..., 0, :, :]
    return np.matmul(A_0, A_0.swapaxes(-1, -2)).real


def _estimate_transfer_function(minimum_phase):
    return np.matmul(minimum_phase,
                     np.linalg.inv(minimum_phase[..., 0:1, :, :]))


def _squared_magnitude(x):
    return np.abs(x) ** 2


def _complex_inner_product(a, b):
    return np.matmul(a, _conjugate_transpose(b))


def _remove_instantaneous_causality(noise_covariance):
    noise_covariance = noise_covariance[..., np.newaxis, :, :]
    variance = np.diagonal(noise_covariance, axis1=-1,
                           axis2=-2)[..., np.newaxis]
    return (_conjugate_transpose(variance) -
            noise_covariance * _conjugate_transpose(noise_covariance) /
            variance)


def _set_diagonal_to_zero(x):
    n_signals = x.shape[-1]
    diagonal_index = np.diag_indices(n_signals)
    x[..., diagonal_index[0], diagonal_index[1]] = 0
    return x


def _total_inflow(transfer_function, noise_variance=1.0):
    return np.sqrt(np.sum(
        noise_variance * _squared_magnitude(transfer_function),
        keepdims=True, axis=-1))


def _get_noise_variance(noise_covariance):
    return np.diagonal(noise_covariance, axis1=-1, axis2=-2)[
        ..., np.newaxis, :, np.newaxis]


def _total_outflow(MVAR_Fourier_coefficients, noise_variance):
    return np.sqrt(np.sum(
        (1.0 / noise_variance) *
        _squared_magnitude(MVAR_Fourier_coefficients),
        keepdims=True, axis=-2))


def _reshape(fourier_coefficients):
    '''Combine trials and tapers dimensions'''
    (n_time_samples, _, n_signals,
     n_fft_samples, _) = fourier_coefficients.shape
    return fourier_coefficients.swapaxes(1, 3).reshape(
        (n_time_samples, n_fft_samples, n_signals, -1))


def _normalize_fourier_coefficients(fourier_coefficients):
    '''Normalizes a group of fourier coefficients by power'''
    U, _, V = np.linalg.svd(
        _reshape(fourier_coefficients), full_matrices=False)
    return np.matmul(U, V)


def _estimate_canonical_coherency(normalized_fourier_coefficients1,
                                  normalized_fourier_coefficients2):
    group_cross_spectrum = _complex_inner_product(
        normalized_fourier_coefficients1, normalized_fourier_coefficients2)
    return np.linalg.svd(group_cross_spectrum,
                         full_matrices=False, compute_uv=False)[..., 0]


def _bandpass(data, frequencies, frequencies_of_interest, axis=-3):
    frequency_index = ((frequencies_of_interest[0] < frequencies) &
                       (frequencies < frequencies_of_interest[1]))
    return (np.take(data, frequency_index.nonzero()[0], axis=axis),
            frequencies[frequency_index])


def _get_independent_frequency_step(frequency_difference,
                                    frequency_resolution):
    '''Find the number of points of a frequency axis such that they
    are statistically independent given a frequency resolution.


    Parameters
    ----------
    frequency_difference : float
        The distance between two frequency points
    frequency_resolution : float
        The ability to resolve frequency points

    Returns
    -------
    frequency_step : int
        The number of points required so that two
        frequency points are statistically independent.
    '''
    return int(np.ceil(frequency_resolution / frequency_difference))


def _find_largest_significant_group(is_significant):
    '''Finds the largest cluster of significant values over frequencies.

    If frequency value is signficant and its neighbor in the next frequency
    is also a signficant value, then they are part of the same cluster.

    If there are two clusters of the same size, the first one encountered
    is the signficant cluster. All other signficant values are set to
    false.

    Parameters
    ----------
    is_significant : bool array

    Returns
    -------
    is_significant_largest : bool array

    '''
    labeled, _ = label(is_significant)
    label_groups, label_counts = np.unique(labeled, return_counts=True)

    if len(label_groups) > 1:
        label_counts[0] = 0
        max_group = label_groups[np.argmax(label_counts)]
        return labeled == max_group
    else:
        return np.zeros(is_significant.shape, dtype=bool)


def _get_independent_frequencies(is_significant, frequency_step):
    '''Given a `frequency_step` that determines the distance to the next
    signficant point, sets non-distinguishable points to false.

    Parameters
    ----------
    is_significant : bool array

    Returns
    -------
    is_significant_independent : bool array

    '''
    index = is_significant.nonzero()[0]
    independent_index = index[slice(0, len(index), frequency_step)]
    return np.in1d(np.arange(0, len(is_significant)), independent_index)


def _find_largest_independent_group(is_significant, frequency_step,
                                    min_group_size=3):
    '''Finds the largest signficant cluster of frequency points and
    returns the indpendent frequency points of that cluster

    Parameters
    ----------
    is_significant : bool array
    frequency_step : int
        The number of points between each independent frequency step
    min_group_size : int
        The minimum number of points for a group to be considered

    Returns
    -------
    is_significant : bool array

    '''
    is_significant = _find_largest_significant_group(is_significant)
    is_significant = _get_independent_frequencies(
        is_significant, frequency_step)
    if sum(is_significant) < min_group_size:
        is_significant[:] = False
    return is_significant


def _find_significant_frequencies(
    coherency, bias, frequency_step=1, significance_threshold=0.05,
    min_group_size=3,
        multiple_comparisons_method='Benjamini_Hochberg_procedure'):
    '''Determines the largest significant cluster along the frequency axis.

    This function uses the fisher z-transform to determine the p-values and
    adjusts for multiple comparisons using the
    `multiple_comparisons_method`. Only independent frequencies are
    returned and there must be at least `min_group_size` frequency
    points for the cluster to be returned. If there are several significant
    groups, then only the largest group is returned.

    Parameters
    ----------
    coherency : array, shape (..., n_frequencies, n_signals, n_signals)
        The complex coherency between signals.
    bias : float
        Bias from the number of indpendent estimates of the frequency
        transform.
    frequency_step : int
        The number of points between each independent frequency step
    significance_threshold : float
        The threshold for a p-value to be considered signficant.
    min_group_size : int
        The minimum number of independent frequency points for
    multiple_comparisons_method : 'Benjamini_Hochberg_procedure' |
                                  'Bonferroni_correction'
        Procedure used to correct for multiple comparisons.

    Returns
    -------
    is_significant : bool array, shape (..., n_frequencies, n_signals,
                                        n_signals)

    '''
    z_coherence = fisher_z_transform(coherency, bias)
    p_values = get_normal_distribution_p_values(z_coherence)
    is_significant = adjust_for_multiple_comparisons(
        p_values, alpha=significance_threshold)
    return np.apply_along_axis(_find_largest_independent_group, -3,
                               is_significant, frequency_step,
                               min_group_size)


def _conjugate_transpose(x):
    '''Conjugate transpose of the last two dimensions of array x'''
    return x.swapaxes(-1, -2).conjugate()
