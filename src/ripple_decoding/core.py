'''Classifying sharp-wave ripple replay events from spiking activity
(e.g. Forward vs. Reverse replay)

References
----------
.. [1] Deng, X., Liu, D.F., Karlsson, M.P., Frank, L.M., and Eden, U.T.
       (2016). Rapid classification of hippocampal replay content for
       real-time applications. Journal of Neurophysiology 116, 2221-2235.

'''
from logging import getLogger

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from statsmodels.api import GLM, families

logger = getLogger(__name__)


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
    return distribution / np.nansum(distribution, axis=-1)


def _get_prior(posterior, state_transition):
    '''The prior given the current posterior density and a transition
    matrix indicating the state at the next time step.
    '''
    return np.matmul(
        state_transition, posterior[..., np.newaxis]).squeeze()


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


def empirical_movement_transition_matrix(place, place_bin_edges,
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
    place = np.array(place)
    movement_bins, _, _ = np.histogram2d(place[1:], place[:-1],
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


def get_bin_centers(bin_edges):
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
    place_bin_size = place_bin_edges[1] - place_bin_edges[0]

    outbound_initial_conditions = normalize_to_probability(
        _normal_pdf(place_bin_centers, mean=0,
                    std_deviation=place_bin_size * 2))

    inbound_initial_conditions = normalize_to_probability(
        (np.max(outbound_initial_conditions) *
         np.ones(place_bin_centers.shape)) -
        outbound_initial_conditions)

    return {'Inbound': inbound_initial_conditions,
            'Outbound': outbound_initial_conditions}
