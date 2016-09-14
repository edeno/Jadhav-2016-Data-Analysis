import os
import scipy.io
import scipy.fftpack
import scipy.signal
import scipy.ndimage
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import spectral
import data_filter as df


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
            .assign(ripple_indicator=lambda x: x.ripple_zscore >= zscore_threshold))


def get_ripple_zscore_multitaper(lfp, sampling_frequency, frequency_resolution=100,
                                 time_window_duration=0.020, zscore_threshold=3):
    ''' Returns a pandas dataframe containing the original lfp and the ripple-band (150-250 Hz)
    score for the lfp using a tapered power signal centered at 200 Hz. Frequency resolution is
    100 Hz and time resolution is 20 milliseconds by default.
    '''
    time_window_step = time_window_duration / 2

    spectrogram = spectral.get_spectrogram_dataframe(lfp,
                                                     frequency_resolution=frequency_resolution,
                                                     time_window_duration=time_window_duration,
                                                     sampling_frequency=sampling_frequency,
                                                     time_window_step=time_window_step)
    is_200_Hz = spectrogram.frequency == 200
    dataframes = [pd.DataFrame({'ripple_zscore': _zscore(spectrogram.loc[is_200_Hz, 'power'])}),
                  lfp.reset_index()]
    return (pd.concat(dataframes, axis=1)
              .set_index('time')
              .assign(ripple_indicator=lambda x: x.ripple_zscore >= zscore_threshold))


def _get_series_start_end_times(series):
    ''' Returns a two element tuple of the start of the segment
    and the end of the segment. The input series must be a boolean
    pandas series where the index is time.
    '''
    is_start_time = (~series.shift(1).fillna(False)) & series
    start_times = series.index[is_start_time]

    is_end_time = series & (~series.shift(-1).fillna(False))
    end_times = series.index[is_end_time]

    # Handle case of the indicator starting or ending above threshold.
    # Remove these cases from the list
    if len(start_times) != len(end_times):
        if end_times[0] > start_times[0]:
            end_times = end_times[1:]
        else:
            start_times = start_times[:-1]

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
