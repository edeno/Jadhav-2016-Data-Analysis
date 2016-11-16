import numpy as np
import pandas as pd
import scipy.ndimage.filters
import scipy.linalg
import scipy.stats
import patsy
import statsmodels.api as sm
import data_processing


def predict_state(data, initial_conditions=None, state_transition=None,
                  likelihood_function=None, likelihood_kwargs={}, debug=False):
    ''' Adaptive filter
    '''
    posterior = initial_conditions
    num_states = len(initial_conditions)
    num_time_points = data.shape[0]
    posterior_over_time = np.zeros((num_time_points, num_states))
    if debug:
        likelihood_over_time = np.zeros((num_time_points, num_states))
        prior_over_time = np.zeros((num_time_points, num_states))
    for time_ind in np.arange(num_time_points):
        posterior_over_time[time_ind, :] = posterior
        prior = _get_prior(posterior, state_transition)
        likelihood = likelihood_function(data[time_ind, :], **likelihood_kwargs)
        posterior = _update_posterior(prior, likelihood)
        if debug:
            likelihood_over_time[time_ind, :] = likelihood
            prior_over_time[time_ind, :] = prior
    if not debug:
        return posterior_over_time
    else:
        return posterior_over_time, likelihood_over_time, prior_over_time


def _update_posterior(prior, likelihood):
    ''' Yields the posterior density given the prior state
    weighted by the observed instantaneous likelihood
    '''
    return normalize_to_probability(prior * likelihood)


def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1
    so that it is a probability distribution
    '''
    return distribution / np.sum(distribution.flatten())


def _get_prior(posterior, state_transition):
    ''' Yields the prior given the current posterior
    density and a transition matrix indicating the
    state at the next time step.
    '''
    return np.dot(state_transition, posterior)


def instantaneous_poisson_likelihood(is_spike, conditional_intensity=None, time_bin_size=1):
    probability_spike = conditional_intensity
    probability_no_spike = np.exp(-conditional_intensity * time_bin_size)
    return (probability_spike ** is_spike) * probability_no_spike


def combined_likelihood(data, likelihood_function=None, likelihood_kwargs={}):
    ''' Combine likelihoods over columns. If there isn't a column dimension,
    just return the likelihood. The likelihood function must take data as its
    first argument. All other arguments for the likelihood should be passed
    via the likelihood keyword argument ('likelihood_kwargs')
    '''
    try:
        return np.prod(likelihood_function(data, **likelihood_kwargs), axis=1)
    except ValueError:
        return likelihood_function(data, **likelihood_kwargs)


def empirical_movement_transition_matrix(linear_position, linear_position_grid,
                                         speed_up_factor=30):
    ''' Estimates the probablity of the next position based on the movement data.
    '''
    movement_bins, _, _ = np.histogram2d(linear_position, linear_position.shift(1),
                                         bins=(linear_position_grid, linear_position_grid),
                                         normed=False)
    movement_bins_probability = _normalize_column_probability(_fix_zero_bins(movement_bins))
    smoothed_movement_bins_probability = scipy.ndimage.filters.gaussian_filter(
        movement_bins_probability, sigma=0.5)
    return _normalize_column_probability(
        np.linalg.matrix_power(smoothed_movement_bins_probability, speed_up_factor))


def _normalize_column_probability(x):
    '''Ensure the state transition matrix columns integrate to 1
    so that it is a probability distribution
    '''
    return np.dot(x, np.diag(1 / np.sum(x, axis=0)))


def _fix_zero_bins(movement_bins):
    ''' If there is no data observed for a column, set everything to 1
    so that it will have equal probability
    '''
    is_zero_column = np.sum(movement_bins, axis=0) == 0
    movement_bins[:, is_zero_column] = 1
    return movement_bins


def decode_ripple(epoch_index, animals, ripple_times,
                  sampling_frequency=1500,
                  linear_distance_grid=np.linspace(0, 192, 49),
                  likelihood_function=instantaneous_poisson_likelihood):
    print('\nDecoding ripples for Animal {0}, Day {1}, Epoch #{2}:'.format(*epoch_index))
    # Include only CA1 neurons with spikes
    neuron_info = data_processing.make_neuron_dataframe(animals)[epoch_index].dropna()
    tetrode_info = data_processing.make_tetrode_dataframe(animals)[epoch_index]
    neuron_info = pd.merge(tetrode_info, neuron_info,
                           on=['animal', 'day', 'epoch_ind', 'tetrode_number', 'area'],
                           how='right', right_index=True).set_index(neuron_info.index)
    neuron_info = neuron_info[neuron_info.area.isin(['CA1', 'iCA1']) &
                              (neuron_info.numspikes > 0) &
                              ~neuron_info.descrip.str.endswith('Ref').fillna(False)]

    # Train on when the rat is moving
    position_info = data_processing.get_interpolated_position_dataframe(
        epoch_index, animals)
    spikes_data = [data_processing.get_spike_indicator_dataframe(neuron_index, animals)
                   for neuron_index in neuron_info.index]

    # Make sure there are spikes in the training data times. Otherwise exclude that neuron
    spikes_data = [spikes_datum for spikes_datum in spikes_data
                   if spikes_datum[position_info.speed > 4].sum().values > 0]

    train_position_info = position_info.query('speed > 4')
    train_spikes_data = [spikes_datum[position_info.speed > 4]
                         for spikes_datum in spikes_data]
    linear_distance_grid_centers = _get_grid_centers(linear_distance_grid)

    # Fit encoding model
    print('\tFitting encoding model...')
    conditional_intensity = get_encoding_model(
        train_position_info, train_spikes_data, linear_distance_grid_centers)

    # Fit state transition model
    print('\tFitting state transition model...')
    state_transition = get_state_transition_matrix(train_position_info, linear_distance_grid)

    # Initial Conditions
    print('\tSetting initial conditions...')
    state_names = ['outbound_forward', 'outbound_reverse',
                   'inbound_forward', 'inbound_reverse']
    num_states = len(state_names)
    initial_conditions = get_initial_conditions(
        linear_distance_grid, linear_distance_grid_centers, num_states)

    # Decode
    print('\tDecoding ripples...')
    combined_likelihood_params = dict(
        likelihood_function=likelihood_function,
        likelihood_kwargs=dict(conditional_intensity=conditional_intensity)
        )
    decoder_params = dict(
        initial_conditions=initial_conditions,
        state_transition=state_transition,
        likelihood_function=combined_likelihood,
        likelihood_kwargs=combined_likelihood_params
    )
    test_spikes = _get_ripple_spikes(spikes_data, ripple_times, sampling_frequency)
    posterior_density = [predict_state(ripple_spikes, **decoder_params)
                         for ripple_spikes in test_spikes]
    session_time = position_info.index
    return get_ripple_info(posterior_density, test_spikes, ripple_times, state_names, session_time)


def _get_ripple_spikes(spikes_data, ripple_times, sampling_frequency):
    spike_ripples_df = [data_processing.reshape_to_segments(
        spikes_datum, ripple_times, concat_axis=1, sampling_frequency=sampling_frequency)
                        for spikes_datum in spikes_data]

    return [np.vstack([df.iloc[:, ripple_ind].dropna().values
                       for df in spike_ripples_df]).T
            for ripple_ind in np.arange(len(ripple_times))]


def _get_grid_centers(grid):
    return grid[:-1] + np.diff(grid) / 2


def get_initial_conditions(linear_distance_grid, linear_distance_grid_centers,
                           num_states):
    linear_distance_grid_bin_size = linear_distance_grid[
        1] - linear_distance_grid[0]

    outbound_initial_conditions = normalize_to_probability(
        scipy.stats.norm.pdf(linear_distance_grid_centers, 0, linear_distance_grid_bin_size * 2))

    inbound_initial_conditions = normalize_to_probability(
        (np.max(outbound_initial_conditions) * np.ones(linear_distance_grid_centers.shape)) -
        outbound_initial_conditions)

    prior_probability_of_state = 1 / num_states
    return np.hstack([outbound_initial_conditions,
                      inbound_initial_conditions,
                      inbound_initial_conditions,
                      outbound_initial_conditions]) * prior_probability_of_state


def get_state_transition_matrix(train_position_info, linear_distance_grid):
    inbound_state_transitions = empirical_movement_transition_matrix(
        train_position_info[train_position_info.trajectory_direction == 'Inbound'].linear_distance,
        linear_distance_grid)
    outbound_state_transitions = empirical_movement_transition_matrix(
        train_position_info[train_position_info.trajectory_direction == 'Outbound'].linear_distance,
        linear_distance_grid)

    return scipy.linalg.block_diag(outbound_state_transitions,
                                   inbound_state_transitions,
                                   inbound_state_transitions,
                                   outbound_state_transitions)


def get_encoding_model(train_position_info, train_spikes_data, linear_distance_grid_centers):
    formula = '1 + trajectory_direction * bs(linear_distance, df=10, degree=3)'
    design_matrix = patsy.dmatrix(
        formula, train_position_info, return_type='dataframe')
    fit = [sm.GLM(spikes, design_matrix, family=sm.families.Poisson()).fit(maxiter=30)
           for spikes in train_spikes_data]

    inbound_predict_design_matrix = _predictors_by_trajectory_direction(
        'Inbound', linear_distance_grid_centers, design_matrix)
    outbound_predict_design_matrix = _predictors_by_trajectory_direction(
        'Outbound', linear_distance_grid_centers, design_matrix)

    inbound_conditional_intensity = _get_conditional_intensity(
        fit, inbound_predict_design_matrix)
    outbound_conditional_intensity = _get_conditional_intensity(
        fit, outbound_predict_design_matrix)

    return np.vstack([outbound_conditional_intensity,
                      outbound_conditional_intensity,
                      inbound_conditional_intensity,
                      inbound_conditional_intensity])


def get_ripple_info(posterior_density, test_spikes, ripple_times, state_names, session_time):
    ''' Summary statistics for ripples
    '''
    num_states = len(state_names)
    num_ripples = len(ripple_times)
    decision_state_probability = [_compute_decision_state_probability(density, num_states)
                                  for density in posterior_density]

    ripple_info = pd.DataFrame([_compute_max_state(probability, state_names)
                                for probability in decision_state_probability],
                               columns=['ripple_trajectory', 'ripple_direction',
                                        'ripple_state_probability'],
                               index=pd.Index(np.arange(num_ripples) + 1, name='ripple_number'))
    ripple_info['ripple_start_time'] = np.asarray(ripple_times)[:, 0]
    ripple_info['ripple_end_time'] = np.asarray(ripple_times)[:, 1]
    ripple_info['number_of_unique_neurons_spiking'] = [_num_unique_neurons_spiking(spikes)
                                                       for spikes in test_spikes]
    ripple_info['number_of_spikes'] = [_num_total_spikes(spikes)
                                       for spikes in test_spikes]
    ripple_info['session_time'] = _ripple_session_time(ripple_times, session_time)
    ripple_info['is_spike'] = ((ripple_info.number_of_spikes > 0)
                               .map({True: 'isSpike', False: 'noSpike'}))

    return ripple_info, decision_state_probability, posterior_density, state_names


def _predictors_by_trajectory_direction(trajectory_direction, linear_distance_grid_centers,
                                        design_matrix):
    predictors = {'linear_distance': linear_distance_grid_centers,
                  'trajectory_direction': [trajectory_direction] *
                  len(linear_distance_grid_centers)}
    return patsy.build_design_matrices([design_matrix.design_info], predictors)[0]


def _get_conditional_intensity(fit, predict_design_matrix):
    return np.vstack([fitted_model.predict(predict_design_matrix)
                      for fitted_model in fit]).T


def _compute_decision_state_probability(posterior_density, num_states):
    num_time = len(posterior_density)
    new_shape = (num_time, num_states, -1)
    return np.sum(np.reshape(posterior_density, new_shape), axis=2)


def _compute_max_state(probability, state_names):
    end_time_probability = probability[-1, :]
    return (*state_names[np.argmax(end_time_probability)].split('_'),
            np.max(end_time_probability))


def _num_unique_neurons_spiking(spikes):
    ''' Number of units that spike per ripple
    '''
    return spikes.sum(axis=0).nonzero()[0].shape[0]


def _num_total_spikes(spikes):
    ''' Total number of spikes per ripple
    '''
    return int(spikes.sum(axis=(0, 1)))


def _ripple_session_time(ripple_times, session_time):
    ''' Categorize the ripples by the time in the session in which they occur.
    This function trichotimizes the session time into early session, middle session,
    and late session and classifies the ripple by the most prevelant category.
    '''
    session_time_categories = pd.Series(pd.cut(
        session_time, 3, labels=['early', 'middle', 'late'], precision=4), index=session_time)
    return [session_time_categories.loc[ripple_start:ripple_end].value_counts().argmax()
            for ripple_start, ripple_end in ripple_times]
