'''Finding sharp-wave ripple events (150-250 Hz) from local field
potentials

'''
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import filtfilt, hilbert
from scipy.stats import zscore

from .spectral.transforms import Multitaper
from .spectral.connectivity import Connectivity


def _get_series_start_end_times(series):
    '''Returns a two element tuple with of the start of the segment and the
     end of the segment. Each element is an numpy array, The input series
    must be a boolean pandas series where the index is time.
    '''
    is_start_time = (~series.shift(1).fillna(False)) & series
    start_times = series.index[is_start_time].get_values()

    is_end_time = series & (~series.shift(-1).fillna(False))
    end_times = series.index[is_end_time].get_values()

    return start_times, end_times


def segment_boolean_series(series, minimum_duration=0.015):
    '''Returns a list of tuples where each tuple contains the start time of
     segement and end time of segment. It takes a boolean pandas series as
     input where the index is time.
     '''
    start_times, end_times = _get_series_start_end_times(series)

    return [(start_time, end_time)
            for start_time, end_time in zip(start_times, end_times)
            if end_time >= (start_time + minimum_duration)]


def multitaper_Kay_method(lfps, minimum_duration=0.015,
                          sampling_frequency=1500, zscore_threshold=2,
                          multitaper_kwargs={}):
    '''Uses the multitaper ripple-band power from each tetrode and combines
    using
    '''
    m = Multitaper(
        pd.Panel(lfps).values.squeeze().T,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=0.020,
        time_window_step=0.020,
        start_time=0)
    c = Connectivity.from_multitaper(m)
    ripple_band_index = (c.frequencies >= 150) & (c.frequencies <= 250)
    ripple_power = np.mean(c.power()[..., ripple_band_index, :], axis=-2)
    return _get_candidate_ripples_Kay(
        ripple_power, is_multitaper=True,
        zscore_threshold=zscore_threshold,
        minimum_duration=minimum_duration)


def mulititaper_Karlsson_method(lfps, minimum_duration=0.015,
                                sampling_frequency=1500,
                                zscore_threshold=2, multitaper_kwargs={}):
    m = Multitaper(
        pd.Panel(lfps).values.squeeze().T,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=0.020,
        time_window_step=0.020,
        start_time=0)
    c = Connectivity.from_multitaper(m)
    ripple_band_index = (c.frequencies >= 150) & (c.frequencies <= 250)
    ripple_power = np.mean(c.power()[..., ripple_band_index, :], axis=-2)
    return _get_candidate_ripples_Karlsson(
        ripple_power, minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold)


def Kay_method(lfps, minimum_duration=0.015, zscore_threshold=2,
               smoothing_sigma=0.004, sampling_frequency=1500):
    filtered_lfps = [pd.Series(
        _ripple_bandpass_filter(lfp.values.ravel()), index=lfp.index)
        for lfp in lfps]
    return _get_candidate_ripples_Kay(
        filtered_lfps, is_multitaper=False,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold, sigma=smoothing_sigma,
        sampling_frequency=sampling_frequency)


def Karlsson_method(lfps, smoothing_sigma=0.004, sampling_frequency=1500,
                    minimum_duration=0.015, zscore_threshold=2):
    ripple_envelope = [_get_smoothed_envelope(lfp, smoothing_sigma,
                                              sampling_frequency)
                       for lfp in lfps]
    return _get_candidate_ripples_Karlsson(
        ripple_envelope, minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold)


def _get_smoothed_envelope(lfp, sigma, sampling_frequency):
    '''Filters the lfp between 150-250 Hz and returns the
    smoothed envelope of the filtered signal
    '''
    return pd.Series(_smooth(_get_envelope(_ripple_bandpass_filter(
        lfp.values.flatten())), sigma, sampling_frequency),
        index=lfp.index)


def _get_candidate_ripples_Kay(filtered_lfps, is_multitaper=False,
                               minimum_duration=0.015,
                               zscore_threshold=2, sigma=0.004,
                               sampling_frequency=1500):
    '''Candidate ripple times are extracted based on mean power on all
    tetrodes

    Using power from all tetrodes reflects that we view the ripple as a
    population event

    Parameters
    ----------
    filtered_lfps : Pandas series or array_like
        Time series of ripple band power
    is_multitaper : bool
        Are we using multi-taper methods to extract the power?
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a candidate ripple
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to be
        considered a candidate ripple
    sigma : float, optional
    sampling_frequency : int, optional

    Returns
    -------
    ripple_times : list of 2-element tuples
        The elements correspond to the start and stop times of the ripple

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
    and Frank, L.M. (2016). A hippocampal network for spatial coding during
    immobility and sleep. Nature 531, 185-190.

    '''
    combined_lfps = _squared_sum(pd.concat(filtered_lfps, axis=1), axis=1)

    smooth_combined_lfps = combined_lfps if is_multitaper else pd.Series(
        _smooth(combined_lfps.values.flatten(), sigma, sampling_frequency),
        index=combined_lfps.index)

    thresholded_combined_lfps = _threshold_by_zscore(
        np.sqrt(smooth_combined_lfps), zscore_threshold=zscore_threshold)
    return list(sorted(_extend_threshold_to_mean(
        thresholded_combined_lfps.is_above_mean,
        thresholded_combined_lfps.is_above_threshold,
        minimum_duration=minimum_duration)))


def _squared_sum(data, axis=1):
    return np.sum(data, axis=axis) ** 2


def _get_candidate_ripples_Karlsson(filtered_lfps, minimum_duration=0.015,
                                    zscore_threshold=2):
    '''Candidate ripple times are detected on each tetrode and then
    combined if they overlap

    Parameters
    ----------
    filtered_lfps : Pandas series or array_like
        Time series of ripple band power
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a candidate ripple
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to be
        considered a candidate ripple

    Returns
    -------
    ripple_times : list of 2-element tuples
        The elements correspond to the start and stop times of the ripple

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
    experiences in the hippocampus. Nature Neuroscience 12, 913-918.

    '''
    thresholded_lfps = [_threshold_by_zscore(
        lfp, zscore_threshold=zscore_threshold)
        for lfp in filtered_lfps]
    candidate_ripple_times = [_extend_threshold_to_mean(
        lfp.is_above_mean, lfp.is_above_threshold,
        minimum_duration=minimum_duration)
        for lfp in thresholded_lfps]
    return list(_merge_overlapping_ranges(
        _flatten_list(candidate_ripple_times)))


def _flatten_list(original_list):
    '''Takes a list of lists and turns it into a single list'''
    return [item for sublist in original_list for item in sublist]


def _ripple_bandpass_filter(data):
    '''Returns a bandpass filtered signal between 150-250 Hz using the
    Frank lab filter
    '''
    filter_numerator, filter_denominator = _get_ripplefilter_kernel()
    is_nan = np.isnan(data)
    filtered_data = np.full_like(data, np.nan)
    filtered_data[~is_nan] = filtfilt(
        filter_numerator, filter_denominator, data[~is_nan], axis=0)
    return filtered_data


def _get_ripplefilter_kernel():
    '''Returns the pre-computed ripple filter kernel from the Frank lab.
    The kernel is 150-250 Hz bandpass with 40 db roll off and 10 Hz
    sidebands.
    '''
    data_dir = '{working_dir}/Raw-Data'.format(
        working_dir=os.path.abspath(os.path.pardir))
    ripplefilter = loadmat(
        '{data_dir}/ripplefilter.mat'.format(data_dir=data_dir))
    return ripplefilter['ripplefilter']['kernel'][0][0].flatten(), 1


def _extend_threshold_to_mean(is_above_mean, is_above_threshold,
                              minimum_duration=0.015):
    '''Extract segments above threshold if they remain above the threshold
    for a minimum amount of time and extend them to the mean

    Parameters
    ----------
    is_above_mean : Pandas series
        Time series indicator function specifying when the
        time series is above the mean. Index of the series is time.
    is_above_threshold : Pandas series
        Time series indicator function specifying when the
        time series is above the the threshold. Index of the series is
        time.

    Returns
    -------
    extended_segments : list of 2-element tuples
        Elements correspond to the start and end time of segments

    '''
    above_mean_segments = segment_boolean_series(
        is_above_mean, minimum_duration=minimum_duration)
    above_threshold_segments = segment_boolean_series(
        is_above_threshold, minimum_duration=minimum_duration)
    return _extend_segment(above_threshold_segments, above_mean_segments)


def _find_containing_interval(interval_candidates, target_interval):
    '''Returns the interval that contains the target interval out of a list
    of interval candidates.

    This is accomplished by finding the closest start time out of the
    candidate intervals, since we already know that one interval candidate
    contains the target interval (the segements above 0 contain the
    segments above the threshold)
    '''
    candidate_start_times = np.asarray(interval_candidates)[:, 0]
    closest_start_ind = np.max(
        (candidate_start_times - target_interval[0] <= 0).nonzero())
    return interval_candidates[closest_start_ind]


def _extend_segment(segments_to_extend, containing_segments):
    '''Extends the boundaries of a segment if it is a subset of one of the
    containing segments.

    Parameters
    ----------
    segments_to_extend : list of 2-element tuples
        Elements are the start and end times
    containing_segments : list of 2-element tuples
        Elements are the start and end times

    Returns
    -------
    extended_segments : list of 2-element tuples

    '''
    segments = [_find_containing_interval(containing_segments, segment)
                for segment in segments_to_extend]
    return list(set(segments))  # remove duplicate segments


def _get_envelope(data, axis=0):
    '''Extracts the instantaneous amplitude (envelope) of an analytic
     signal using the Hilbert transform'''
    return np.abs(hilbert(data, axis=axis))


def _smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    '''1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    '''
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis)


def _threshold_by_zscore(data, zscore_threshold=2):
    '''Standardize the data and determine whether it is above a given
    number.

    Parameters
    ----------
    data : array_like or Pandas series
    zscore_threshold : int, optional

    Returns
    -------
    threshold_dataframe : Pandas dataframe
        Dataframe contains two columns. One column is an indicator function
        where 1 indicates the z-score is above the mean (z-score = 0) and 0
        indicates the z-score is below the mean. The other column is an
        indicator function of z-scored data where 1 indicates the z-score
        is above the threshold parameter and 0 indicates the z-score is
        below the threshold parameter.

    '''
    zscored_data = zscore(data)
    return pd.DataFrame(
        {'is_above_threshold': zscored_data >= zscore_threshold,
         'is_above_mean': zscored_data >= 0}, index=data.index)


def _merge_overlapping_ranges(ranges):
    '''Merge overlapping and adjacent ranges

    Parameters
    ----------
    ranges : iterable with 2-elements
        Element 1 is the start of the range.
        Element 2 is the end of the range.
    Yields
    -------
    sorted_merged_range : 2-element tuple
        Element 1 is the start of the merged range.
        Element 2 is the end of the merged range.

    >>> list(_merge_overlapping_ranges([(5,7), (3,5), (-1,3)]))
    [(-1, 7)]
    >>> list(_merge_overlapping_ranges([(5,6), (3,4), (1,2)]))
    [(1, 2), (3, 4), (5, 6)]
    >>> list(_merge_overlapping_ranges([]))
    []

    References
    ----------
    .. [1] http://codereview.stackexchange.com/questions/21307/consolidate-
    list-of-ranges-that-overlap

    '''
    ranges = iter(sorted(ranges))
    current_start, current_stop = next(ranges)
    for start, stop in ranges:
        if start > current_stop:
            # Gap between segments: output current segment and start a new
            # one.
            yield current_start, current_stop
            current_start, current_stop = start, stop
        else:
            # Segments adjacent or overlapping: merge.
            current_stop = max(current_stop, stop)
    yield current_start, current_stop
