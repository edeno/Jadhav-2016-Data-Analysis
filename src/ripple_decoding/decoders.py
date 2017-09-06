from logging import getLogger

import numpy as np

import xarray as xr

from .clusterless import (build_joint_mark_intensity,
                          estimate_ground_process_intensity,
                          poisson_mark_likelihood)
from .core import (combined_likelihood, empirical_movement_transition_matrix,
                   get_bin_centers, predict_state, set_initial_conditions)

logger = getLogger(__name__)

_DEFAULT_STATE_NAMES = ['Outbound-Forward', 'Outbound-Reverse',
                        'Inbound-Forward', 'Inbound-Reverse']

_DEFAULT_OBSERVATION_STATE_ORDER = ['Outbound', 'Outbound',
                                    'Inbound', 'Inbound']

_DEFAULT_STATE_TRANSITION_STATE_ORDER = ['Outbound', 'Inbound',
                                         'Inbound', 'Outbound']


class ClusterlessDecoder(object):
    '''

    Attributes
    ----------
    position : ndarray, shape (n_time,)
        Position of the animal to train the model on.
    trajectory_direction : array_like, shape (n_time,)
        Task of the animal. Element must be either
         'Inbound' or 'Outbound'.
    spike_marks : ndarray, shape (n_time, n_marks, n_signals)
        Marks to train the model on.
        If spike does not occur, the row must be marked with np.nan
    n_position_bins : int, optional
    mark_std_deviation : float, optional

    '''

    def __init__(self, position, trajectory_direction, spike_marks,
                 n_position_bins=61, mark_std_deviation=20,
                 replay_speedup_factor=16,
                 state_names=_DEFAULT_STATE_NAMES,
                 observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
                 state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER):
        self.position = np.array(position)
        self.trajectory_direction = np.array(trajectory_direction)
        self.spike_marks = np.array(spike_marks)
        self.n_position_bins = n_position_bins
        self.mark_std_deviation = mark_std_deviation
        self.replay_speedup_factor = replay_speedup_factor
        self.state_names = state_names
        self.observation_state_order = observation_state_order
        self.state_transition_state_order = state_transition_state_order

    def fit(self):
        '''Fits the decoder model for each trajectory_direction.

        Relates the position and spike_marks to the trajectory_direction.

        Parameters
        ----------


        Returns
        -------
        self : class instance

        '''

        self.place_bin_edges = np.linspace(
            np.floor(self.position.min()), np.ceil(self.position.max()),
            self.n_position_bins + 1)
        self.place_std_deviation = np.diff(self.place_bin_edges)[0]
        self.place_bin_centers = get_bin_centers(self.place_bin_edges)

        initial_conditions = set_initial_conditions(
            self.place_bin_edges, self.place_bin_centers)
        initial_conditions = np.stack(
            [initial_conditions[state]
             for state in self.state_transition_state_order]
        ) / len(self.state_names)
        self.initial_conditions = xr.DataArray(
            initial_conditions, dims=['state', 'position'],
            coords=dict(position=self.place_bin_centers,
                        state=self.state_names),
            name='initial_conditions')

        trajectory_directions = np.unique(self.trajectory_direction)

        logger.info('Fitting state transition model...')

        state_transition_by_state = {
            direction: empirical_movement_transition_matrix(
                self.position[
                    np.in1d(self.trajectory_direction, direction)],
                self.place_bin_edges, self.replay_speedup_factor)
            for direction in trajectory_directions}
        state_transition_matrix = np.stack(
            [state_transition_by_state[state]
             for state in self.state_transition_state_order])
        self.state_transition_matrix = xr.DataArray(
            state_transition_matrix,
            dims=['state', 'position_t', 'position_t_1'],
            coords=dict(state=self.state_names,
                        position_t=self.place_bin_centers,
                        position_t_1=self.place_bin_centers),
            name='state_transition_probability')

        logger.info('Fitting observation model...')
        joint_mark_intensity_functions = []
        ground_process_intensity = []

        for marks in self.spike_marks:
            jmi_by_state = {
                direction: build_joint_mark_intensity(
                    self.position[
                        np.in1d(self.trajectory_direction, direction)],
                    marks[np.in1d(self.trajectory_direction, direction)],
                    self.place_bin_centers, self.place_std_deviation,
                    self.mark_std_deviation)
                for direction in trajectory_directions}
            joint_mark_intensity_functions.append(
                [jmi_by_state[state]
                 for state in self.observation_state_order])

            gpi_by_state = {
                direction: estimate_ground_process_intensity(
                    self.position[
                        np.in1d(self.trajectory_direction, direction)],
                    marks[np.in1d(self.trajectory_direction, direction)],
                    self.place_bin_centers, self.place_std_deviation)
                for direction in trajectory_directions}
            ground_process_intensity.append(
                [gpi_by_state[state]
                 for state in self.observation_state_order])

        ground_process_intensity = np.stack(ground_process_intensity)
        likelihood_kwargs = dict(
            joint_mark_intensity_functions=joint_mark_intensity_functions,
            ground_process_intensity=ground_process_intensity)

        self._combined_likelihood_kwargs = dict(
            likelihood_function=poisson_mark_likelihood,
            likelihood_kwargs=likelihood_kwargs)

        return self

    def predict(self, spike_marks, time=None):
        '''Predicts the state from spike_marks.

        Parameters
        ----------
        spike_marks : ndarray, shape (n_time, n_marks)
            If spike does not occur, the row must be marked with np.nan.
        time : ndarray, optional, shape (n_time,)

        Returns
        -------
        predicted_state : str

        '''
        posterior_density = predict_state(
            spike_marks,
            initial_conditions=self.initial_conditions.values,
            state_transition=self.state_transition_matrix.values,
            likelihood_function=combined_likelihood,
            likelihood_kwargs=self._combined_likelihood_kwargs)
        coords = dict(
            time=(time if time is not None
                  else np.arange(posterior_density.shape[0])),
            position=self.place_bin_centers,
            state=self.state_names
        )

        posterior_density = xr.DataArray(
            posterior_density,
            dims=['time', 'state', 'position'],
            coords=coords,
            name='posterior_density')

        return DecodingResults(
            posterior_density=posterior_density,
            test_marks=spike_marks
        )


class SortedSpikeDecoder(object):

    def __init__(self, position, spikes, trajectory_direction,
                 n_position_bins=61):
        '''

        Attributes
        ----------
        position : ndarray, shape (n_time,)
        spike : ndarray, shape (n_time,)
        trajectory_direction : ndarray, shape (n_time,)
        n_position_bins : int, optional

        '''
        self.n_position_bins = n_position_bins

    def fit(self):
        '''Fits the decoder model by state

        Relates the position and spikes to the state.
        '''
        return self

    def predict(self, spikes):
        '''Predicts the state from the spikes.

        Parameters
        ----------
        spike : ndarray, shape (n_time,)

        Returns
        -------
        predicted_state : str

        '''
        pass


class DecodingResults():

    def __init__(self, posterior_density, test_marks=None):
        self.posterior_density = posterior_density
        self.test_marks = test_marks

    def state_probability(self):
        return self.posterior_density.sum('position').to_series().unstack()

    def predicted_state(self):
        return self.state_probability().iloc[-1].argmax()

    def predicted_state_probability(self):
        return self.state_probability().iloc[-1].max()

    def spike_train(self):
        return np.any(~np.isnan(self.test_marks), axis=-1) * 1
