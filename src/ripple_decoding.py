'''Classifying sharp-wave ripple replay events from spiking activity
(e.g. Forward vs. Reverse replay)

'''

from functools import partial
from warnings import warn

import numpy as np
import pandas as pd
from numba import jit
from patsy import build_design_matrices, dmatrix
from scipy.linalg import block_diag
from scipy.ndimage.filters import gaussian_filter
from statsmodels.api import GLM, families

from src.data_processing import (get_interpolated_position_dataframe,
                                 get_spike_indicator_dataframe,
                                 make_neuron_dataframe, make_tetrode_dataframe,
                                 reshape_to_segments)


def predict_state(data, initial_conditions=None, state_transition=None,
                  likelihood_function=None, likelihood_kwargs={},
                  debug=False):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable

    Parameters
    ----------
    data : array_like, shape=(n_time, n_signals, ...)
    initial_conditions : array_like (n_parameters * n_states,)
    state_transition : array_like (n_parameters * n_states,
                                   n_parameters * n_states)
    likelihood_function : function
    likelihood_kwargs: dict, optional
        Additional arguments to the likelihood function
        besides the data
    debug : bool, optional
        If true, function returns likelihood and prior

    Returns
    -------
    posterior_over_time : array_like, shape=(n_time_points,
                                             n_parameters * n_states)
    likelihood_over_time : array_like, shape=(n_time_points,
                                              n_parameters * n_states)
    prior_over_time : array_like, shape=(n_time_points,
                                         n_parameters * n_states)

    '''
    posterior = initial_conditions
    n_parameters = initial_conditions.shape[0]
    n_time_points = data.shape[0]
    posterior_over_time = np.zeros((n_time_points, n_parameters))
    if debug:
        likelihood_over_time = np.zeros((n_time_points, n_parameters))
        prior_over_time = np.zeros((n_time_points, n_parameters))
    for time_ind in np.arange(n_time_points):
        posterior_over_time[time_ind, :] = posterior
        prior = _get_prior(posterior, state_transition)
        likelihood = likelihood_function(
            data[time_ind, ...], **likelihood_kwargs)
        posterior = _update_posterior(prior, likelihood)
        if debug:
            likelihood_over_time[time_ind, :] = likelihood
            prior_over_time[time_ind, :] = prior
    if not debug:
        return posterior_over_time
    else:
        return posterior_over_time, likelihood_over_time, prior_over_time


def _update_posterior(prior, likelihood):
    '''The posterior density given the prior state weighted by the
    observed instantaneous likelihood
    '''
    return normalize_to_probability(prior * likelihood)


def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / distribution.sum()


def _get_prior(posterior, state_transition):
    '''The prior given the current posterior density and a transition
    matrix indicating the state at the next time step.
    '''
    return np.dot(state_transition, posterior)


def poisson_likelihood(is_spike, conditional_intensity=None,
                       time_bin_size=1):
    '''Probability of parameters given spiking at a particular time

    Parameters
    ----------
    is_spike : array_like with values of {0, 1}, shape=(n_signals,)
        Indicator of spike or no spike at current time.
    conditional_intensity : array_like, shape=(n_time_points, n_signals)
        Instantaneous probability of observing a spike
    time_bin_size : float, optional

    Returns
    -------
    poisson_likelihood : array_like, shape=(n_parameters,)

    '''
    probability_no_spike = np.exp(-conditional_intensity * time_bin_size)
    return ((conditional_intensity ** is_spike[:, np.newaxis]) *
            probability_no_spike)


def poisson_mark_likelihood(marks, joint_mark_intensity=None,
                            ground_process_intensity=None,
                            time_bin_size=1):
    '''Probability of parameters given spiking indicator at a particular
    time and associated marks.

    Parameters
    ----------
    marks : array_like, shape=(n_signals, n_marks)
    joint_mark_intensity : function
        Instantaneous probability of observing a spike given mark vector
        from data. The parameters for this function should already be set,
        before it is passed to `poisson_mark_likelihood`.
    ground_process_intensity : array_like, shape=(n_signals,
                                                  n_parameters * n_states)
        Probability of observing a spike regardless of marks.
    time_bin_size : float, optional

    Returns
    -------
    poisson_mark_likelihood : array_like, shape=(n_signals, n_parameters)

    '''
    is_spike = np.all(~np.isnan(marks), axis=1).flatten()
    probability_no_spike = np.exp(-ground_process_intensity *
                                  time_bin_size)
    return (joint_mark_intensity(marks) ** is_spike[:, np.newaxis] *
            probability_no_spike)


def evaluate_mark_space(test_marks, training_marks=None,
                        mark_std_deviation=20):
    '''Evaluate the multivariate Gaussian kernel for the mark space
    given training marks.

    For each mark in the training data (`training_marks`), a univariate
    Gaussian is placed with its mean at the value of each mark with
    standard deviation `mark_std_deviation`. The product of the Gaussians
    along the mark dimension yields a multivariate Gaussian kernel
    evaluated at each training spike with a diagonal coviarance matrix.

    Parameters
    ----------
    test_marks : array_like, shape=(n_marks,)
        The marks to be evaluated
    training_marks : shape=(n_training_spikes, n_marks)
        The marks for each spike when the animal is moving
    mark_std_deviation : float, optional
        The standard deviation of the Gaussian kernel in millivolts

    Returns
    -------
    mark_space_estimator : array_like, shape=(n_training_spikes,)

    '''
    n_training_spikes = training_marks.shape[0]
    test_marks = np.tile(
        test_marks[:, np.newaxis], (1, n_training_spikes)).T
    return np.nanprod(
        _normal_pdf(test_marks, mean=training_marks,
                    std_deviation=mark_std_deviation),
        axis=1)


def joint_mark_intensity(marks, place_field_estimator=None,
                         place_occupancy=None,
                         training_marks=None,
                         mark_std_deviation=20):
    '''Evaluate the multivariate density function of the marks and place
    field for each signal

    Parameters
    ----------
    marks : array_like, shape=(n_signals, n_marks)
    place_field_estimator : n_signal-element list of arrays of
                            shape=(n_parameters, n_training_spikes)
    place_occupancy : array_like, shape=(n_parameters,)
        The probability that the animal is at that position
    training_marks : n_signal-element list of arrays of
                     shape=(n_training_spikes, n_marks)
        The marks for each spike when the animal is moving
    mark_std_deviation : float, optional
        The standard deviation of the Gaussian kernel in millivolts

    Returns
    -------
    joint_mark_intensity : array_like, shape=(n_signals, n_parameters)

    '''

    n_parameters = place_occupancy.shape[0]
    n_signals = len(place_field_estimator)
    place_mark_estimator = np.zeros((n_signals, n_parameters))

    for signal_ind in range(n_signals):
        place_mark_estimator[signal_ind, :] = np.dot(
            place_field_estimator[signal_ind],
            evaluate_mark_space(
                marks[signal_ind],
                training_marks=training_marks[signal_ind],
                mark_std_deviation=mark_std_deviation)
        )

    return place_mark_estimator / place_occupancy


def estimate_place_field(place_bin_centers, place_at_spike,
                         place_std_deviation=1):
    '''Non-parametric estimate of the neuron receptive field with respect
    to place.

    Puts a Gaussian with a mean at the position the animal is located at
    when there is a spike

    Parameters
    ----------
    place_bin_centers : array_like, shape=(n_parameters,)
        Evaluate the Gaussian at these bins
    place_at_spike : array_like, shape=(n_training_spikes,)
        Position of the animal at spike time
    place_std_deviation : float, optional
        Standard deviation of the Gaussian kernel

    Returns
    -------
    place_field_estimator : array_like, shape=(n_parameters,
                                               n_training_spikes)

    '''
    n_parameters, n_spikes = (place_bin_centers.shape[0],
                              place_at_spike.shape[0])
    place_bin_centers = np.tile(place_bin_centers[:, np.newaxis],
                                (1, n_spikes))
    place_at_spike = np.tile(
        place_at_spike[:, np.newaxis], (1, n_parameters)).T
    return _normal_pdf(place_bin_centers, mean=place_at_spike,
                       std_deviation=place_std_deviation)


def estimate_ground_process_intensity(place_field_estimator,
                                      place_occupancy):
    '''The probability of observing a spike regardless of mark

    Parameters
    ----------
    place_field_estimator : array_like, shape=(n_parameters * n_states,
                                               n_training_spikes)
    place_occupancy : array_like, shape=(n_parameters * n_states,)

    Returns
    -------
    ground_process_intensity : array_like, shape=(n_parameters * n_states,)

    '''
    return normalize_to_probability(
        place_field_estimator.sum(axis=1) / place_occupancy)


def estimate_place_occupancy(place_bin_centers, place,
                             place_std_deviation=1):
    '''A Gaussian smoothed probability that the animal is in a particular
    position.

    Denominator in equation #12 and #13 of [1]

    Parameters
    ----------
    place_bin_centers : array_like, shape=(n_parameters,)
    place : array_like, shape=(n_places,)
    place_std_deviation : float, optional

    Returns
    -------
    place_occupancy : array_like, shape=(n_parameters,)

    '''
    n_parameters, n_places = place_bin_centers.shape[0], place.shape[0]
    place_bin_centers = np.tile(place_bin_centers[:, np.newaxis],
                                (1, n_places))
    place = np.tile(place[:, np.newaxis], (1, n_parameters)).T
    return _normal_pdf(place_bin_centers, mean=place,
                       std_deviation=place_std_deviation).sum(axis=1)


def estimate_marked_encoding_model(place_bin_centers, place,
                                   place_at_spike, training_marks,
                                   place_std_deviation=4,
                                   mark_std_deviation=20):
    '''Non-parametric estimatation of place fields based on marks

    A Gaussian kernel is placed at each mark and place the animal is at
    when a spike occurs.

    Parameters
    ----------
    place : list, n_states
    place_at_spike : list of lists of arrays, n_signals * n_states
    place_bin_centers : array_like, shape=(n_parameters,)
    training_marks : list of lists of arrays, n_signals * n_states
    place_std_deviation : float, optional

    Returns
    -------
    combined_likelihood_kwargs : dict
        Keyword arguments for the `combined_likelihood`
        function.

    '''
    n_signals, n_states = len(place_at_spike), len(place)

    place_occupancy = [
        estimate_place_occupancy(
            place_bin_centers, place[state_ind],
            place_std_deviation=place_std_deviation)
        for state_ind in range(n_states)]

    ground_process_intensity = list()
    place_field_estimator = list()
    marks = list()

    for signal_ind in range(n_signals):
        signal_place_field = [
            estimate_place_field(
                place_bin_centers, place_at_spike[signal_ind][state_ind],
                place_std_deviation=place_std_deviation)
            for state_ind in range(n_states)]

        signal_ground_process_intensity = [
            estimate_ground_process_intensity(
                signal_place_field[state_ind], place_occupancy[state_ind])
            for state_ind in range(n_states)]

        place_field_estimator.append(
            block_diag(*signal_place_field))
        ground_process_intensity.append(
            np.hstack(signal_ground_process_intensity))
        marks.append(np.vstack(training_marks[signal_ind]))

    place_occupancy = np.hstack(place_occupancy)
    ground_process_intensity = np.stack(ground_process_intensity)

    fixed_joint_mark_intensity = partial(
        joint_mark_intensity, place_field_estimator=place_field_estimator,
        place_occupancy=place_occupancy, training_marks=training_marks,
        mark_std_deviation=mark_std_deviation)

    return dict(
        likelihood_function=poisson_mark_likelihood,
        likelihood_kwargs=dict(
            joint_mark_intensity=fixed_joint_mark_intensity,
            ground_process_intensity=ground_process_intensity)
    )


def combined_likelihood(data, likelihood_function=None,
                        likelihood_kwargs={}):
    '''Applies likelihood function to each signal and returns their product

    If there isn't a column dimension, just returns the likelihood.

    Parameters
    ----------
    data : array_like, shape=(n_signals, ...)
    likelihood_function : function
        Likelihood function to be applied to each signal.
        The likelihood function must take data as its first argument.
        All other arguments for the likelihood should be passed
        via `likelihood_kwargs`
    likelihood_kwargs : dict
        Keyword arguments for the likelihood function

    Returns
    -------
    likelihood : array_like, shape=(n_parameters * n_states,)

    '''
    try:
        return np.nanprod(
            likelihood_function(data, **likelihood_kwargs),
            axis=0).squeeze()
    except ValueError:
        return likelihood_function(data, **likelihood_kwargs).squeeze()


def empirical_movement_transition_matrix(place,
                                         place_bin_edges,
                                         sequence_compression_factor=16):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `sequence_compression_factor`

    Place cell firing during a hippocampal replay event is a "sped-up"
    version of place cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which place
    cell is likely to fire next.

    Parameters
    ----------
    place : array_like
        Linearized position of the animal over time
    place_bin_edges : array_like
    sequence_compression_factor : int, optional
        How much the movement is sped-up during a replay event

    Returns
    -------
    empirical_movement_transition_matrix : array_like,
                                           shape=(n_bin_edges-1,
                                           n_bin_edges-1)

    '''
    movement_bins, _, _ = np.histogram2d(place[1:],
                                         place[:-1],
                                         bins=(place_bin_edges,
                                               place_bin_edges),
                                         normed=False)
    smoothed_movement_bins_probability = gaussian_filter(
        _normalize_column_probability(
            _fix_zero_bins(movement_bins)), sigma=0.5)
    return _normalize_column_probability(
        np.linalg.matrix_power(
            smoothed_movement_bins_probability,
            sequence_compression_factor))


def _normalize_column_probability(x):
    '''Ensure the state transition matrix columns integrate to 1
    so that it is a probability distribution
    '''
    return np.dot(x, np.diag(1 / x.sum(axis=0)))


def _fix_zero_bins(movement_bins):
    '''If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    '''
    movement_bins[:, movement_bins.sum(axis=0) == 0] = 1
    return movement_bins


def decode_ripple(epoch_index, animals, ripple_times,
                  sampling_frequency=1500,
                  n_place_bins=49):
    '''Labels the ripple by category

    Parameters
    ----------
    epoch_index : 3-element tuple
        Specifies which epoch to run.
        (Animal short name, day, epoch_number)
    animals : list of named-tuples
        Tuples give information to convert from the animal short name
        to a data directory
    ripple_times : list of 2-element tuples
        The first element of the tuple is the start time of the ripple.
        Second element of the tuple is the end time of the ripple
    sampling_frequency : int, optional
        Sampling frequency of the spikes
    n_place_bins : int, optional
        Number of bins for the linear distance

    Returns
    -------
    ripple_info : pandas dataframe
        Dataframe containing the categories for each ripple
        and the probability of that category

    '''
    print('\nDecoding ripples for Animal {0}, Day {1}, Epoch #{2}:'.format(
        *epoch_index))
    # Include only CA1 neurons with spikes
    neuron_info = make_neuron_dataframe(animals)[
        epoch_index].dropna()
    tetrode_info = make_tetrode_dataframe(animals)[
        epoch_index]
    neuron_info = pd.merge(tetrode_info, neuron_info,
                           on=['animal', 'day', 'epoch_ind',
                               'tetrode_number', 'area'],
                           how='right', right_index=True).set_index(
        neuron_info.index)
    neuron_info = neuron_info[
        neuron_info.area.isin(['CA1', 'iCA1']) &
        (neuron_info.numspikes > 0) &
        ~neuron_info.descrip.str.endswith('Ref').fillna(False)]

    # Train on when the rat is moving
    position_info = get_interpolated_position_dataframe(
        epoch_index, animals)
    spikes_data = [get_spike_indicator_dataframe(neuron_index, animals)
                   for neuron_index in neuron_info.index]

    # Make sure there are spikes in the training data times. Otherwise
    # exclude that neuron
    spikes_data = [spikes_datum for spikes_datum in spikes_data
                   if spikes_datum[
                       position_info.speed > 4].sum().values > 0]

    train_position_info = position_info.query('speed > 4')
    train_spikes_data = [spikes_datum[position_info.speed > 4]
                         for spikes_datum in spikes_data]
    place_bin_edges = np.linspace(
        np.floor(position_info.linear_distance.min()),
        np.ceil(position_info.linear_distance.max()),
        n_place_bins + 1)
    place_bin_centers = _get_bin_centers(
        place_bin_edges)

    print('\tFitting encoding model...')
    combined_likelihood_kwargs = estimate_sorted_spike_encoding_model(
        train_position_info, train_spikes_data, place_bin_centers)

    print('\tFitting state transition model...')
    state_transition = estimate_state_transition(
        train_position_info, place_bin_edges)

    print('\tSetting initial conditions...')
    state_names = ['outbound_forward', 'outbound_reverse',
                   'inbound_forward', 'inbound_reverse']
    n_states = len(state_names)
    initial_conditions = set_initial_conditions(
        place_bin_edges, place_bin_centers, n_states)

    print('\tDecoding ripples...')
    decoder_params = dict(
        initial_conditions=initial_conditions,
        state_transition=state_transition,
        likelihood_function=combined_likelihood,
        likelihood_kwargs=combined_likelihood_kwargs
    )
    test_spikes = _get_ripple_spikes(
        spikes_data, ripple_times, sampling_frequency)
    posterior_density = [predict_state(ripple_spikes, **decoder_params)
                         for ripple_spikes in test_spikes]
    return get_ripple_info(
        posterior_density, test_spikes, ripple_times,
        state_names, position_info.index)


def _get_ripple_spikes(spikes_data, ripple_times, sampling_frequency):
    '''Given the ripple times, extract the spikes within the ripple
    '''
    spike_ripples_df = [reshape_to_segments(
        spikes_datum, ripple_times,
        concat_axis=1, sampling_frequency=sampling_frequency)
        for spikes_datum in spikes_data]

    return [np.vstack([df.iloc[:, ripple_ind].dropna().values
                       for df in spike_ripples_df]).T
            for ripple_ind in np.arange(len(ripple_times))]


def _get_bin_centers(bin_edges):
    '''Given the outer-points of bins, find their center
    '''
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def set_initial_conditions(place_bin_edges,
                           place_bin_centers, n_states=4):
    '''Sets the prior for each state (Outbound-Forward, Outbound-Reverse,
    Inbound-Forward, Inbound-Reverse).

    Inbound states have greater weight on starting at the center arm.
    Outbound states have weight everywhere else.

    Parameters
    ----------
    place_bin_edges : array_like, shape=(n_parameters,)
        Histogram bin edges of the place measure
    place_bin_centers : array_like, shape=(n_parameters,)
        Histogram bin centers of the place measure
    n_states : int, optional

    Returns
    -------
    initial_conditions : array_like, shape=(n_parameters * n_states,)
        Initial conditions for each state are stacked row-wise.
    '''
    place_bin_size = place_bin_edges[
        1] - place_bin_edges[0]

    outbound_initial_conditions = normalize_to_probability(
        _normal_pdf(place_bin_centers, mean=0,
                    std_deviation=place_bin_size * 2))

    inbound_initial_conditions = normalize_to_probability(
        (np.max(outbound_initial_conditions) *
         np.ones(place_bin_centers.shape)) -
        outbound_initial_conditions)

    prior_probability_of_state = 1 / n_states
    return (np.hstack([outbound_initial_conditions,
                       inbound_initial_conditions,
                       inbound_initial_conditions,
                       outbound_initial_conditions]) *
            prior_probability_of_state)


def estimate_state_transition(train_position_info,
                              place_bin_edges):
    '''The block-diagonal empirical state transition matrix for each state:
    Outbound-Forward, Outbound-Reverse, Inbound-Forward, Inbound-Reverse

    Parameters
    ----------
    train_position_info : pandas dataframe
        The animal's linear distance from the center well
        for each trajectory direction while the animal is moving
    place_bin_edges : array_like, shape=(n_bins+1,)
        bin endpoints to partition the linear distances

    Returns
    -------
    state_transition_matrix : array_like

    '''
    inbound_state_transitions = empirical_movement_transition_matrix(
        (train_position_info[
            train_position_info.trajectory_direction == 'Inbound']
            .linear_distance.values),
        place_bin_edges)
    outbound_state_transitions = empirical_movement_transition_matrix(
        (train_position_info[
            train_position_info.trajectory_direction == 'Outbound']
            .linear_distance.values),
        place_bin_edges)

    return block_diag(outbound_state_transitions,
                      inbound_state_transitions,
                      inbound_state_transitions,
                      outbound_state_transitions)


def glm_fit(spikes, design_matrix, ind):
    '''Fits the Poisson model to the spikes from a neuron

    Parameters
    ----------
    spikes : array_like
    design_matrix : array_like or pandas DataFrame
    ind : int

    Returns
    -------
    fitted_model : object or NaN
        Returns the statsmodel object if successful. If the model fails in
        the weighted fit in the IRLS procedure, the model returns NaN.

    '''
    try:
        print('\t\t...Neuron #{}'.format(ind + 1))
        return GLM(spikes.reindex(design_matrix.index), design_matrix,
                   family=families.Poisson(),
                   drop='missing').fit(maxiter=30)
    except np.linalg.linalg.LinAlgError:
        warn('Data is poorly scaled for neuron #{}'.format(ind + 1))
        return np.nan


def estimate_sorted_spike_encoding_model(train_position_info,
                                         train_spikes_data,
                                         place_bin_centers):
    '''The conditional intensities for each state (Outbound-Forward,
    Outbound-Reverse, Inbound-Forward, Inbound-Reverse)

    Parameters
    ----------
    train_position_info : pandas dataframe
    train_spikes_data : array_like
    place_bin_centers : array_like, shape=(n_parameters,)

    Returns
    -------
    combined_likelihood_kwargs : dict

    '''
    formula = ('1 + trajectory_direction * '
               'bs(linear_distance, df=10, degree=3)')
    design_matrix = dmatrix(
        formula, train_position_info, return_type='dataframe')
    fit = [glm_fit(spikes, design_matrix, ind)
           for ind, spikes in enumerate(train_spikes_data)]

    inbound_predict_design_matrix = _predictors_by_trajectory_direction(
        'Inbound', place_bin_centers, design_matrix)
    outbound_predict_design_matrix = _predictors_by_trajectory_direction(
        'Outbound', place_bin_centers, design_matrix)

    inbound_conditional_intensity = _get_conditional_intensity(
        fit, inbound_predict_design_matrix)
    outbound_conditional_intensity = _get_conditional_intensity(
        fit, outbound_predict_design_matrix)

    conditional_intensity = np.vstack(
        [outbound_conditional_intensity,
         outbound_conditional_intensity,
         inbound_conditional_intensity,
         inbound_conditional_intensity]).T

    return dict(
        likelihood_function=poisson_likelihood,
        likelihood_kwargs=dict(
            conditional_intensity=conditional_intensity)
    )


def get_ripple_info(posterior_density, test_spikes, ripple_times,
                    state_names, session_time):
    '''Summary statistics for ripple categories

    Parameters
    ----------
    posterior_density : array_like
    test_spikes : array_like
    ripple_times : list of tuples
    state_names : list of str
    session_time : array_like

    Returns
    -------
    ripple_info : pandas dataframe
    decision_state_probability : array_like
    posterior_density : array_like
    state_names : list of str

    '''
    n_states = len(state_names)
    n_ripples = len(ripple_times)
    decision_state_probability = [
        _compute_decision_state_probability(density, n_states)
        for density in posterior_density]

    ripple_info = pd.DataFrame(
        [_compute_max_state(probability, state_names)
         for probability in decision_state_probability],
        columns=['ripple_trajectory', 'ripple_direction',
                 'ripple_state_probability'],
        index=pd.Index(np.arange(n_ripples) + 1, name='ripple_number'))
    ripple_info['ripple_start_time'] = np.asarray(ripple_times)[:, 0]
    ripple_info['ripple_end_time'] = np.asarray(ripple_times)[:, 1]
    ripple_info['number_of_unique_neurons_spiking'] = [
        _num_unique_neurons_spiking(spikes) for spikes in test_spikes]
    ripple_info['number_of_spikes'] = [_num_total_spikes(spikes)
                                       for spikes in test_spikes]
    ripple_info['session_time'] = _ripple_session_time(
        ripple_times, session_time)
    ripple_info['is_spike'] = ((ripple_info.number_of_spikes > 0)
                               .map({True: 'isSpike', False: 'noSpike'}))

    return (ripple_info, decision_state_probability,
            posterior_density, state_names)


def _predictors_by_trajectory_direction(trajectory_direction,
                                        place_bin_centers,
                                        design_matrix):
    '''The design matrix for a given trajectory direction
    '''
    predictors = {'linear_distance': place_bin_centers,
                  'trajectory_direction': [trajectory_direction] *
                  len(place_bin_centers)}
    return build_design_matrices(
        [design_matrix.design_info], predictors)[0]


def glm_val(fitted_model, predict_design_matrix):
    '''Predict the model's response given a design matrix
    and the model parameters
    '''
    try:
        return fitted_model.predict(predict_design_matrix)
    except AttributeError:
        return np.ones(predict_design_matrix.shape[0]) * np.nan


def _get_conditional_intensity(fit, predict_design_matrix):
    '''The conditional intensity for each model
    '''
    return np.vstack([glm_val(fitted_model, predict_design_matrix)
                      for fitted_model in fit]).T


def _compute_decision_state_probability(posterior_density, n_states):
    '''The marginal probability of a state given the posterior_density
    '''
    n_time = len(posterior_density)
    new_shape = (n_time, n_states, -1)
    return np.sum(np.reshape(posterior_density, new_shape), axis=2)


def _compute_max_state(probability, state_names):
    '''The discrete state with the highest probability at the last time
    '''
    end_time_probability = probability[-1, :]
    return (*state_names[np.argmax(end_time_probability)].split('_'),
            np.max(end_time_probability))


def _num_unique_neurons_spiking(spikes):
    '''Number of units that spike per ripple
    '''
    return spikes.sum(axis=0).nonzero()[0].shape[0]


def _num_total_spikes(spikes):
    '''Total number of spikes per ripple
    '''
    return int(spikes.sum(axis=(0, 1)))


def _ripple_session_time(ripple_times, session_time):
    '''Categorize the ripples by the time in the session in which they
    occur.

    This function trichotimizes the session time into early session,
    middle session, and late session and classifies the ripple by the most
    prevelant category.
    '''
    session_time_categories = pd.Series(
        pd.cut(
            session_time, 3,
            labels=['early', 'middle', 'late'], precision=4),
        index=session_time)
    return [(session_time_categories
             .loc[ripple_start:ripple_end]
             .value_counts()
             .argmax())
            for ripple_start, ripple_end in ripple_times]


@jit(nopython=True)
def _normal_pdf(x, mean=0, std_deviation=1):
    '''Evaluate the normal probability density function at specified points.

    Unlike the `scipy.norm.pdf`, this function is not general and does not
    do any sanity checking of the inputs. As a result it is a much faster
    function, but you should be sure of your inputs before using.

    Parameters
    ----------
    x : array_like
        The normal probability function will be evaluated
    mean : float or array_like, optional
    std_deviation : float or array_like

    Returns
    -------
    probability_density
        The normal probability density function evaluated at `x`

    '''
    u = (x - mean) / std_deviation
    return np.exp(-0.5 * u ** 2) / (np.sqrt(2.0 * np.pi) * std_deviation)
