class ClusterlessDecoder(object):
    '''

    Attributes
    ----------
    n_position_bins : int, optional
    mark_std_deviation : float, optional

    '''

    def __init__(self, n_position_bins=61, mark_std_deviation=20):
        self.n_position_bins = n_position_bins
        self.mark_std_deviation = mark_std_deviation

    def fit(self, position, spike_marks, states):
        '''Fits the decoder model by state.

        Relates the position and spike_marks to the state.

        Parameters
        ----------
        position : array_like, shape (n_time,)
        spike_marks : array_like, shape (n_time, n_marks)
            If spike does not occur, the row must be marked with np.nan.
        states : array_like, shape (n_time,)

        '''
        return self

    def predict(self, spike_marks):
        '''Predicts the state from spike_marks.

        Parameters
        ----------
        spike_marks : array_like, shape (n_time, n_marks)
            If spike does not occur, the row must be marked with np.nan.

        Returns
        -------
        predicted_state : str

        '''
        pass


class SortedSpikeDecoder(object):

    def __init__(self, n_position_bins=61):
        '''

        Attributes
        ----------
        n_position_bins : int, optional

        '''
        self.n_position_bins = n_position_bins

    def fit(self, position, spikes, states):
        '''Fits the decoder model by state

        Relates the position and spikes to the state.

        Parameters
        ----------
        position : array_like, shape (n_time,)
        spike : array_like, shape (n_time,)
        states : array_like, shape (n_time,)

        '''
        return self

    def predict(self, spikes):
        '''Predicts the state from the spikes.

        Parameters
        ----------
        spike : array_like, shape (n_time,)

        Returns
        -------
        predicted_state : str

        '''
        pass
