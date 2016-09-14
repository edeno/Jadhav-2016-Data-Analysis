import scipy.io
import scipy.fftpack
import scipy.signal
import scipy.ndimage
import numpy as np
import nitime.algorithms as tsa
import matplotlib.pyplot as plt
import pandas as pd
import spectral
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
