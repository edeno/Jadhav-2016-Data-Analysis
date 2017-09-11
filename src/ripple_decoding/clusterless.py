'''Classifying sharp-wave ripple replay events from spiking activity
(e.g. Forward vs. Reverse replay)

References
----------
.. [1] Deng, X., Liu, D.F., Karlsson, M.P., Frank, L.M., and Eden, U.T.
       (2016). Rapid classification of hippocampal replay content for
       real-time applications. Journal of Neurophysiology 116, 2221-2235.

'''

import numpy as np
from numba import jit
from functools import partial


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


def _gaussian_kernel(data, means, std_deviation=1):
    return _normal_pdf(
        data[:, np.newaxis], mean=means, std_deviation=std_deviation)


def poisson_mark_likelihood(marks, joint_mark_intensity_functions=None,
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
    ground_process_intensity : array_like, shape=(n_signals, n_states,
                                                  n_place_bins)
        Probability of observing a spike regardless of marks.
    time_bin_size : float, optional

    Returns
    -------
    poisson_mark_likelihood : array_like, shape=(n_signals, n_place_bins)

    '''
    probability_no_spike = np.exp(-ground_process_intensity *
                                  time_bin_size)
    joint_mark_intensity = np.array(
        [[func(signal_marks) for func in functions_by_state]
         for signal_marks, functions_by_state
         in zip(marks, joint_mark_intensity_functions)])
    return joint_mark_intensity * probability_no_spike


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
    return np.nanprod(
        _normal_pdf(test_marks, mean=training_marks,
                    std_deviation=mark_std_deviation), axis=1)


def joint_mark_intensity(marks, training_marks=None,
                         mark_std_deviation=None,
                         place_field=None, place_occupancy=None):
    '''Evaluate the multivariate density function of the marks and place
    field for each signal

    Parameters
    ----------
    marks : array_like, shape=(n_marks,)
    place_field : array_like, shape=(n_parameters, n_training_spikes)
    place_occupancy : array_like, shape=(n_place_bins,)
        The probability that the animal is at that position
    training_marks : array_like, shape=(n_training_spikes, n_marks)
        The marks for each spike when the animal is moving
    mark_std_deviation : float, optional
        The standard deviation of the Gaussian kernel in millivolts

    Returns
    -------
    joint_mark_intensity : array_like, shape=(n_place_bins,)

    '''
    is_spike = np.any(~np.isnan(marks))
    if is_spike:
        mark_space = evaluate_mark_space(
            marks, training_marks=training_marks,
            mark_std_deviation=mark_std_deviation)
        place_mark_estimator = np.dot(place_field, mark_space)
        return place_mark_estimator / place_occupancy
    else:
        return np.ones(place_occupancy.shape)


def build_joint_mark_intensity(position, training_marks, place_bin_centers,
                               place_std_deviation, mark_std_deviation):
    is_spike = np.any(~np.isnan(training_marks), axis=1)
    place_occupancy = estimate_place_occupancy(
        position, place_bin_centers, place_std_deviation)
    place_field = estimate_place_field(
        position[is_spike], place_bin_centers, place_std_deviation)

    return partial(
        joint_mark_intensity,
        training_marks=training_marks[is_spike, :],
        mark_std_deviation=mark_std_deviation,
        place_occupancy=place_occupancy,
        place_field=place_field
    )


def estimate_place_field(place_at_spike, place_bin_centers,
                         place_std_deviation=1):
    '''Non-parametric estimate of the neuron receptive field with respect
    to place.

    Puts a Gaussian with a mean at the position the animal is located at
    when there is a spike

    Parameters
    ----------
    place_at_spike : array_like, shape=(n_training_spikes,)
        Position of the animal at spike time
    place_bin_centers : array_like, shape=(n_parameters,)
        Evaluate the Gaussian at these bins
    place_std_deviation : float, optional
        Standard deviation of the Gaussian kernel

    Returns
    -------
    place_field_estimator : array_like, shape=(n_parameters,
                                               n_training_spikes)

    '''
    return _gaussian_kernel(
        place_bin_centers, means=place_at_spike,
        std_deviation=place_std_deviation)


def estimate_ground_process_intensity(position, marks, place_bin_centers,
                                      place_std_deviation):
    '''The probability of observing a spike regardless of mark. Marginalize
    the joint mark intensity over the mark space.

    Parameters
    ----------
    place_field : array_like, shape=(n_states, n_training_spikes,
                                     n_parameters)
    place_occupancy : array_like, shape=(n_states, n_place_bins)

    Returns
    -------
    ground_process_intensity : array_like, shape=(n_states, n_place_bins)

    '''
    is_spike = np.any(~np.isnan(marks), axis=1)
    place_field = estimate_place_field(
        position[is_spike], place_bin_centers, place_std_deviation=1)
    place_occupancy = estimate_place_occupancy(
        position, place_bin_centers, place_std_deviation)
    return place_field.sum(axis=1) / place_occupancy


def estimate_place_occupancy(place, place_bin_centers,
                             place_std_deviation=1):
    '''A Gaussian smoothed probability that the animal is in a particular
    position.

    Denominator in equation #12 and #13 of [1]

    Parameters
    ----------
    place : array_like, shape=(n_places,)
    place_bin_centers : array_like, shape=(n_parameters,)
    place_std_deviation : float, optional

    Returns
    -------
    place_occupancy : array_like, shape=(n_parameters,)

    '''
    return _gaussian_kernel(
        place_bin_centers, means=place,
        std_deviation=place_std_deviation).sum(axis=1)


def estimate_marginalized_joint_mark_intensity(
    mark_bin_edges, place_bin_edges, marks, position_at_spike,
        all_positions, mark_std_deviation, place_std_deviation):

    mark_at_spike = _gaussian_kernel(mark_bin_edges, marks,
                                     mark_std_deviation)
    place_at_spike = _gaussian_kernel(place_bin_edges, position_at_spike,
                                      place_std_deviation)
    place_occupancy = _gaussian_kernel(place_bin_edges, all_positions,
                                       place_std_deviation).sum(axis=1)
    return (np.dot(place_at_spike, mark_at_spike.T) /
            place_occupancy[:, np.newaxis])
