import os
import scipy.io
import scipy.fftpack
import scipy.signal
import scipy.ndimage
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import src.spectral as spectral
import src.data_filter as df


def _equiripple_bandpass(lowcut, highcut, sampling_frequency, transition_width=10, num_taps=318):

    edges = [0,
             lowcut - transition_width,
             lowcut, highcut,
             highcut + transition_width,
             0.5 * sampling_frequency]

    b = scipy.signal.remez(num_taps, edges, [0, 1, 0], Hz=sampling_frequency)
    return b, 1


def get_ripplefilter_kernel():
    ''' Returns the pre-computed ripple filter kernel from the Frank lab. The kernel is 150-250 Hz
    bandpass with 40 db roll off and 10 Hz sidebands.
    '''
    data_dir = '{working_dir}/Raw-Data'.format(working_dir=os.path.abspath(os.path.pardir))
    ripplefilter = scipy.io.loadmat('{data_dir}/ripplefilter.mat'.format(data_dir=data_dir))
    return ripplefilter['ripplefilter']['kernel'][0][0].flatten(), 1


def _bandpass_filter(data):
    ''' Returns a bandpass filtered signal ('data') between lowcut and highcut
    '''
    filter_numerator, filter_denominator = get_ripplefilter_kernel()
    return scipy.signal.filtfilt(filter_numerator, filter_denominator, data)


def _zscore(x):
    ''' Returns an array of the z-score of x
    '''
    return (x - x.mean()) / x.std()


def get_ripple_zscore_frank(lfp, sampling_frequency, sigma=0.004, zscore_threshold=3):
    ''' Returns a pandas dataframe containing the original lfp and the ripple-band (150-250 Hz)
    score for the lfp according to Karlsson, M.P., Frank, L.M., 2009. Awake replay of remote
    experiences in the hippocampus. Nature Neuroscience 12, 913–918. doi:10.1038/nn.2344
    '''
    filtered_data = _bandpass_filter(lfp['electric_potential'])
    filtered_data_envelope = abs(scipy.signal.hilbert(filtered_data))
    smoothed_envelope = scipy.ndimage.filters.gaussian_filter1d(filtered_data_envelope,
                                                                sigma * sampling_frequency,
                                                                truncate=8)
    dataframes = [pd.DataFrame({'ripple_zscore': _zscore(smoothed_envelope)}), lfp.reset_index()]
    return (pd.concat(dataframes, axis=1)
            .set_index('time')
            .assign(is_above_ripple_threshold=lambda x: x.ripple_zscore >= zscore_threshold)
            .assign(is_above_ripple_mean=lambda x: x.ripple_zscore >= 0))


def get_ripple_zscore_multitaper(lfp, sampling_frequency, time_halfbandwidth_product=1,
                                 time_window_duration=0.020, zscore_threshold=3,
                                 time_window_step=0.004):
    ''' Returns a pandas dataframe containing the original lfp and the ripple-band (150-250 Hz)
    score for the lfp using a tapered power signal centered at 200 Hz. Frequency resolution is
    100 Hz and time resolution is 20 milliseconds by default.
    '''
    spectrogram = spectral.get_spectrogram_dataframe(lfp,
                                                     time_halfbandwidth_product=time_halfbandwidth_product,
                                                     time_window_duration=time_window_duration,
                                                     sampling_frequency=sampling_frequency,
                                                     time_window_step=time_window_step,
                                                     desired_frequencies=[150, 250],
                                                     pad=None)
    is_200_Hz = spectrogram.frequency == 200
    return (spectrogram.loc[is_200_Hz, :].drop('frequency', axis=1)
                                         .set_index('time').assign(ripple_zscore=lambda x: _zscore(x.power))
                                         .assign(is_above_ripple_threshold=lambda x: x.ripple_zscore >= zscore_threshold)
                                         .assign(is_above_ripple_mean=lambda x: x.ripple_zscore >= 0).sort_index())


def _get_computed_ripple_times(tetrode_tuple, animals):
    ''' Returns a list of tuples for a given tetrode in the format
    (ripple_number, start_index, end_index). The indexes are relative
    to the trial time for that session. Data is extracted from the ripples
    data structure and calculated according to the Frank Lab criterion.
    '''
    animal, day, epoch_ind, tetrode_number = tetrode_tuple
    ripples_data = df.get_data_structure(animals[animal], day, 'ripples', 'ripples')
    start_end = zip(ripples_data[epoch_ind - 1][0][tetrode_number - 1]['starttime'][0, 0].flatten(),
                    ripples_data[epoch_ind - 1][0][tetrode_number - 1]['endtime'][0, 0].flatten())
    return [(ripple_ind + 1, start_time, end_time)
            for ripple_ind, (start_time, end_time) in enumerate(start_end)]


def _convert_ripple_times_to_dataframe(ripple_times, dataframe):
    ''' Given a list of ripple time tuples (ripple #, start time, end time) and a dataframe with a
    time index (such as the lfp dataframe), returns a pandas dataframe with a column with the
    timestamps of each ripple labeled according to the ripple number. Non-ripple times are marked
    as NaN.
    '''
    index_dataframe = dataframe.drop(dataframe.columns, axis=1)
    ripple_dataframe = (pd.concat([index_dataframe.loc[start_time:end_time].assign(ripple_number=number)
                                   for number, start_time, end_time in ripple_times]))
    return pd.concat([dataframe, ripple_dataframe], axis=1, join_axes=[dataframe.index])


def get_computed_ripples_dataframe(tetrode_index, animals):
    ''' Given a tetrode index (animal, day, epoch, tetrode #), returns a pandas dataframe
    with the pre-computed ripples from the Frank lab labeled according to the ripple number.
    Non-ripple times are marked as NaN.
    '''
    ripple_times = _get_computed_ripple_times(tetrode_index, animals)
    lfp_dataframe = df._get_LFP_dataframe(tetrode_index, animals)
    return (_convert_ripple_times_to_dataframe(ripple_times, lfp_dataframe)
            .assign(ripple_indicator=lambda x: x.ripple_number.fillna(0) > 0))


def _get_series_start_end_times(series):
    ''' Returns a two element tuple of the start of the segment
    and the end of the segment. The input series must be a boolean
    pandas series where the index is time.
    '''
    is_start_time = (~series.shift(1).fillna(False)) & series
    start_times = series.index[is_start_time].get_values()

    is_end_time = series & (~series.shift(-1).fillna(False))
    end_times = series.index[is_end_time].get_values()

    return start_times, end_times


def segment_boolean_series(series, minimum_duration=0.015):
    ''' Returns a list of tuples where each tuple contains the
    start time of segement and end time of segment. It takes
    a boolean pandas series as input where the index is time.

    '''
    start_times, end_times = _get_series_start_end_times(series)

    return [(start_time, end_time)
            for start_time, end_time in zip(start_times, end_times)
            if end_time >= (start_time + minimum_duration)]


def create_box(segment, y_low=-5, height=10, alpha=0.3, color='grey'):
    ''' Convenience function for marking ripple times on a figure. Returns a patches rectangle
    object.
    '''
    return patches.Rectangle((segment[0], y_low),
                             segment[1] - segment[0],
                             height,
                             alpha=alpha, color=color)


def _find_containing_interval(interval_candidates, target_interval):
    '''Returns the interval that contains the target interval out of a list of
    interval candidates. This is accomplished by finding the closest start time
    out of the candidate intervals, since we already know that one interval candidate
    contains the target interval (the segements above 0 contain the segments above
    the threshold)'''
    candidate_start_times = np.asarray(interval_candidates)[:, 0]
    closest_start_ind = np.max((candidate_start_times - target_interval[0] <= 0).nonzero())
    return interval_candidates[closest_start_ind]


def extend_segment_intervals(ripple_above_threshold_segments, ripple_above_mean_segments):
    ''' Returns a list of tuples that extend the
    boundaries of the segments by the ripple threshold (i.e ripple z-score > 3)
    to the boundaries of a containing interval defined by when the z-score
    crosses the mean.
    '''
    segments = [_find_containing_interval(ripple_above_mean_segments, segment)
                for segment in ripple_above_threshold_segments]
    return list(set(segments))  # remove duplicate segments


def get_segments_frank(lfp_dataframe, sampling_frequency, zscore_threshold=3, sigma=0.004,
                       minimum_duration=0.015):
    ''' Returns a list of tuples that correspond to the
    start and end of the ripple using the method of Loren Frank's lab.
    '''
    ripple_frank_df = get_ripple_zscore_frank(lfp_dataframe,
                                              sampling_frequency,
                                              zscore_threshold=zscore_threshold,
                                              sigma=sigma)
    ripple_above_mean_segments = segment_boolean_series(ripple_frank_df.is_above_ripple_mean,
                                                        minimum_duration=minimum_duration)
    ripple_above_threshold_segments = segment_boolean_series(ripple_frank_df.is_above_ripple_threshold,
                                                             minimum_duration=minimum_duration)
    return extend_segment_intervals(ripple_above_threshold_segments, ripple_above_mean_segments)


def get_segments_multitaper(lfp_dataframe, sampling_frequency, zscore_threshold=3,
                            minimum_duration=0.015):
    ''' Returns a list of tuples that correspond to the start and end of the ripple using the
    zscore of a taper at 200 Hz to extract the ripples.
    '''
    ripple_frank_df = get_ripple_zscore_multitaper(lfp_dataframe,
                                                   sampling_frequency,
                                                   zscore_threshold=zscore_threshold)
    ripple_above_mean_segments = segment_boolean_series(ripple_frank_df.is_above_ripple_mean,
                                                        minimum_duration=minimum_duration)
    ripple_above_threshold_segments = segment_boolean_series(ripple_frank_df.is_above_ripple_threshold,
                                                             minimum_duration=minimum_duration)
    return extend_segment_intervals(ripple_above_threshold_segments, ripple_above_mean_segments)


def get_multitaper_ripples_dataframe(tetrode_index, animals, sampling_frequency,
                                     zscore_threshold=3, minimum_duration=0.015):
    ''' Given a tetrode index (animal, day, epoch, tetrode #), returns a pandas dataframe
    with the pre-computed ripples using multitapers labeled according to the ripple number.
    Non-ripple times are marked as NaN.
    '''
    lfp_dataframe = df._get_LFP_dataframe(tetrode_index, animals)
    segments = get_segments_multitaper(lfp_dataframe, sampling_frequency,
                                       zscore_threshold=zscore_threshold,
                                       minimum_duration=minimum_duration)
    ripple_times = [(ind + 1, start_time, end_time)
                    for ind, (start_time, end_time) in enumerate(segments)]
    return (_convert_ripple_times_to_dataframe(ripple_times, lfp_dataframe)
            .assign(ripple_indicator=lambda x: x.ripple_number.fillna(0) > 0))
