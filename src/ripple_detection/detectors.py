class KayDetector(object):
    '''

    Attributes
    ----------
    time : array_like, shape (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float, optional
        Number of samples per second. If None, the sampling frequency will
        be inferred from the time. Note that this can less accurate
        because of floating point differences.

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
    and Frank, L.M. (2016). A hippocampal network for spatial coding during
    immobility and sleep. Nature 531, 185-190.

    '''
    def __init__(self, time, LFPs, speed, sampling_frequency=None):
        self.time = time
        self.LFPs = LFPs
        self.speed = speed
        self.sampling_frequency = (sampling_frequency
                                   if sampling_frequency is not None
                                   else 1 / (time[1] - time[0]))

    def detect(self, speed_threshold=4.0, minimum_duration=0.015,
               zscore_threshold=2.0, smoothing_sigma=0.004):
        '''

        Parameters
        ----------
        speed_threshold : float, optional
            Maximum running speed of animal for a ripple
        minimum_duration : float, optional
            Minimum time the z-score has to stay above threshold to be
            considered a ripple. The default is given assuming time is in
            units of seconds.
        zscore_threshold : float, optional
            Number of standard deviations the ripple power must exceed to
            be considered a ripple
        smoothing_sigma : float, optional
            Amount to smooth the time series over time. The default is
            given assuming time is in units of seconds.
        '''
        pass


class KarlssonDetector(object):
    '''

    Attributes
    ----------
    time : array_like, shape (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float, optional
        Number of samples per second. If None, the sampling frequency will
        be inferred from the time. Note that this can less accurate
        because of floating point differences.

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
    experiences in the hippocampus. Nature Neuroscience 12, 913-918.

    '''
    def __init__(self, time, LFPs, speed, sampling_frequency=None,
                 speed_threshold=4.0, minimum_duration=0.015,
                 zscore_threshold=2.0, smoothing_sigma=0.004):
        self.time = time
        self.LFPs = LFPs
        self.speed = speed
        self.sampling_frequency = (sampling_frequency
                                   if sampling_frequency is not None
                                   else 1 / (time[1] - time[0]))
        self.speed_threshold = speed_threshold
        self.minimum_duration = minimum_duration
        self.zscore_threshold = zscore_threshold
        self.minimum_duration = minimum_duration

    def detect(self, speed_threshold=4.0, minimum_duration=0.015,
               zscore_threshold=2.0, smoothing_sigma=0.004):
        '''

        Parameters
        ----------
        speed_threshold : float, optional
            Maximum running speed of animal for a ripple
        minimum_duration : float, optional
            Minimum time the z-score has to stay above threshold to be
            considered a ripple. The default is given assuming time is in
            units of seconds.
        zscore_threshold : float, optional
            Number of standard deviations the ripple power must exceed to
            be considered a ripple
        smoothing_sigma : float, optional
            Amount to smooth the time series over time. The default is
            given assuming time is in units of seconds.
        '''
        pass


class LongTaoDetector(object):
    def __init__(self, time, LFPs=None, speed=None, spikes=None,
                 sampling_frequency=None):
        pass
