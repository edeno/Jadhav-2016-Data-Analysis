import numpy as np
import scipy.ndimage.filters


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
    return _normalize_to_probability(prior * likelihood)


def _normalize_to_probability(distribution):
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


def empirical_movement_transition_matrix(linear_position, linear_position_grid):
    ''' Estimates the probablity of the next position based on the movement data.
    '''
    movement_bins, _, _ = np.histogram2d(linear_position, linear_position.shift(1),
                                         bins=(linear_position_grid, linear_position_grid),
                                         normed=False)
    movement_bins_probability = _normalize_column_probability(_fix_zero_bins(movement_bins))
    smoothed_movement_bins_probability = scipy.ndimage.filters.gaussian_filter(
        movement_bins_probability, sigma=0.5)
    return _normalize_column_probability(smoothed_movement_bins_probability)


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
