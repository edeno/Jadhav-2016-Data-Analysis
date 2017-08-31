class ClusterlessDecoder(object):
    def __init__(self, n_place_bins=61, mark_std_deviation=20):
        self.n_place_bins = n_place_bins
        self.mark_std_deviation = mark_std_deviation

    def fit(self, position, spike_marks, states):
        pass

    def predict(self, spike_marks):
        pass


class SortedSpikeDecoder(object):
    def __init__(self, n_place_bins=61):
        self.n_place_bins = n_place_bins

    def fit(self, position, spikes, states):
        pass

    def predict(self, spikes):
        pass
