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

from src.data_processing import (get_data_structure,
                                 get_interpolated_position_dataframe,
                                 get_LFP_dataframe, make_tetrode_dataframe)
from src.spectral import multitaper_spectrogram


def _get_computed_ripple_times(tetrode_tuple, animals):
    '''Returns a list of tuples for a given tetrode in the format
    (start_index, end_index). The indexes are relative
    to the trial time for that session. Data is extracted from the ripples
    data structure and calculated according to the Frank Lab criterion.
    '''
    animal, day, epoch_ind, tetrode_number = tetrode_tuple
    ripples_data = get_data_structure(
        animals[animal], day, 'ripples', 'ripples')
    return zip(
        ripples_data[epoch_ind - 1][0][tetrode_number -
                                       1]['starttime'][0, 0].flatten(),
        ripples_data[epoch_ind - 1][0][tetrode_number
                                       - 1]['endtime'][0, 0].flatten())


def get_computed_consensus_ripple_times(epoch_index, animals):
    '''Returns a list of tuples for a given epoch in the format
    (start_time, end_time).
    '''
    animal, day, epoch_ind = epoch_index
    ripples_data = get_data_structure(
        animals[animal], day, 'candripples', 'candripples')
    return list(map(tuple, ripples_data[epoch_ind - 1]['riptimes'][0][0]))


def get_computed_ripples_dataframe(tetrode_index, animals):
    '''Given a tetrode index (animal, day, epoch, tetrode #), returns a
    pandas dataframe with the pre-computed ripples from the Frank lab
     labeled according to the ripple number. Non-ripple times are marked as
     NaN.
    '''
    ripple_times = _get_computed_ripple_times(tetrode_index, animals)
    [(ripple_ind + 1, start_time, end_time)
     for ripple_ind, (start_time, end_time) in enumerate(ripple_times)]
    lfp_dataframe = get_LFP_dataframe(
        tetrode_index, animals)
    return (_convert_ripple_times_to_dataframe(ripple_times, lfp_dataframe)
            .assign(
                ripple_indicator=lambda x: x.ripple_number.fillna(0) > 0))


def _convert_ripple_times_to_dataframe(ripple_times, dataframe):
    '''Given a list of ripple time tuples (ripple #, start time, end time)
    and a dataframe with a time index (such as the lfp dataframe), returns
    a pandas dataframe with a column with the timestamps of each ripple
    labeled according to the ripple number. Non-ripple times are marked as
    NaN.
    '''
    try:
        index_dataframe = dataframe.drop(dataframe.columns, axis=1)
    except AttributeError:
        index_dataframe = dataframe[0].drop(dataframe[0].columns, axis=1)
    ripple_dataframe = (pd.concat(
        [index_dataframe.loc[start_time:end_time].assign(
            ripple_number=number)
         for number, start_time, end_time in ripple_times]))
    try:
        ripple_dataframe = pd.concat(
            [dataframe, ripple_dataframe], axis=1,
            join_axes=[index_dataframe.index])
    except TypeError:
        ripple_dataframe = pd.concat(
            [pd.concat(dataframe, axis=1), ripple_dataframe],
            axis=1, join_axes=[index_dataframe.index])
    return ripple_dataframe


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
    ripple_power = [_get_ripple_power_multitaper(lfp, sampling_frequency,
                                                 **multitaper_kwargs)
                    for lfp in lfps]
    return _get_candidate_ripples_Kay(
        ripple_power, is_multitaper=True,
        zscore_threshold=zscore_threshold,
        minimum_duration=minimum_duration)


def mulititaper_Karlsson_method(lfps, minimum_duration=0.015,
                                sampling_frequency=1500,
                                zscore_threshold=2, multitaper_kwargs={}):
    ripple_power = [_get_ripple_power_multitaper(lfp, sampling_frequency,
                                                 **multitaper_kwargs)
                    for lfp in lfps]
    return _get_candidate_ripples_Karlsson(
        ripple_power, minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold)


def Kay_method(lfps, minimum_duration=0.015, zscore_threshold=2,
               smoothing_sigma=0.004, sampling_frequency=1500):
    filtered_lfps = [pd.Series(
        _ripple_bandpass_filter(lfp.values.flatten()), index=lfp.index)
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


def get_epoch_ripples(epoch_index, animals, sampling_frequency,
                      ripple_detection_function=Kay_method,
                      ripple_detection_kwargs={}, speed_threshold=4):
    '''Returns a list of tuples containing the start and end times of
    ripples. Candidate ripples are computed via the ripple detection
    function and then filtered to exclude ripples where the animal was
    still moving.
    '''
    print('\nDetecting ripples for Animal {0}, Day {1}, Epoch #{2}...\n'.format(
        *epoch_index))
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_index]
    # Get cell-layer CA1, iCA1 LFPs
    area_critera = (tetrode_info.area.isin(['CA1', 'iCA1']) &
                    tetrode_info.descrip.isin(['riptet']))
    tetrode_indices = tetrode_info[area_critera].index.tolist()
    CA1_lfps = [get_LFP_dataframe(tetrode_index, animals)
                for tetrode_index in tetrode_indices]
    candidate_ripple_times = ripple_detection_function(
        CA1_lfps, **ripple_detection_kwargs)
    return _exclude_movement_during_ripples(
        candidate_ripple_times, epoch_index, animals, speed_threshold)


def _exclude_movement_during_ripples(ripple_times, epoch_index, animals,
                                     speed_threshold):
    '''Excludes ripples where the head direction speed is greater than the
    speed threshold. Only looks at the start of the ripple to determine
    head movement speed for the ripple.
    '''
    position_df = get_interpolated_position_dataframe(
        epoch_index, animals)
    return [(ripple_start, ripple_end)
            for ripple_start, ripple_end in ripple_times
            if position_df.loc[
                ripple_start:ripple_end].speed.iloc[0] < speed_threshold]


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
    combined_lfps = np.sum(pd.concat(filtered_lfps, axis=1) ** 2, axis=1)

    if not is_multitaper:
        smooth_combined_lfps = pd.Series(
            _smooth(combined_lfps.values.flatten(),
                    sigma, sampling_frequency),
            index=combined_lfps.index)
    else:
        smooth_combined_lfps = combined_lfps

    threshold_df = _threshold_by_zscore(np.sqrt(smooth_combined_lfps),
                                        zscore_threshold=zscore_threshold)
    return list(sorted(_extend_threshold_to_mean(
        threshold_df.is_above_mean, threshold_df.is_above_threshold,
        minimum_duration=minimum_duration)))


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
    .. [1] Karlsson, M.P., and Frank, L.M. (2009).
    Awake replay of remote experiences in the hippocampus.
    Nature Neuroscience 12, 913-918.

    '''
    thresholded_lfps = [_threshold_by_zscore(
        lfp, zscore_threshold=zscore_threshold)
        for lfp in filtered_lfps]
    extended_lfps = [_extend_threshold_to_mean(
        threshold_df.is_above_mean, threshold_df.is_above_threshold,
        minimum_duration=minimum_duration)
        for threshold_df in thresholded_lfps]
    return list(_merge_ranges(_flatten_list(extended_lfps)))


def _flatten_list(original_list):
    '''Takes a list of lists and turns it into a single list'''
    return [item for sublist in original_list for item in sublist]


def _get_ripple_power_multitaper(lfp, sampling_frequency,
                                 time_halfbandwidth_product=1,
                                 time_window_duration=0.020,
                                 time_window_step=0.004):
    '''Extracts the smoothed spectral power time course
    in the ripple frequency band (150-250 Hz)

    Parameters
    ----------
    lfp : Pandas series or array_like
        Local field electric potential with time as the index
    sampling_frequency : int

    Returns
    -------
    ripple_power : Pandas series

    '''
    return multitaper_spectrogram(
        lfp,
        time_halfbandwidth_product=time_halfbandwidth_product,
        time_window_duration=time_window_duration,
        sampling_frequency=sampling_frequency,
        time_window_step=time_window_step,
        desired_frequencies=[150, 250],
        pad=None).loc[200, :]


def _ripple_bandpass_filter(data):
    '''Returns a bandpass filtered signal between 150-250 Hz using the
    Frank lab filter
    '''
    filter_numerator, filter_denominator = _get_ripplefilter_kernel()
    return filtfilt(filter_numerator, filter_denominator, data, axis=0)


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


def _merge_ranges(ranges):
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

    >>> list(_merge_ranges([(5,7), (3,5), (-1,3)]))
    [(-1, 7)]
    >>> list(_merge_ranges([(5,6), (3,4), (1,2)]))
    [(1, 2), (3, 4), (5, 6)]
    >>> list(_merge_ranges([]))
    []

    References
    ----------
    .. [1] http://codereview.stackexchange.com/questions/21307/consolidate-list-of-ranges-that-overlap

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
