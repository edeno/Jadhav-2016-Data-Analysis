from logging import getLogger
from warnings import warn

import numpy as np
from patsy import build_design_matrices, dmatrix
from statsmodels.api import GLM, families

logger = getLogger(__name__)


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
        logger.debug('\t\t...Neuron #{}'.format(ind + 1))
        fit = GLM(spikes.reindex(design_matrix.index), design_matrix,
                  family=families.Poisson(),
                  drop='missing').fit(maxiter=30)
        return fit if fit.converged else np.nan
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
        return np.full(predict_design_matrix.shape[0], np.nan)


def _get_conditional_intensity(fit, predict_design_matrix):
    '''The conditional intensity for each model
    '''
    return np.vstack([glm_val(fitted_model, predict_design_matrix)
                      for fitted_model in fit]).T


def poisson_likelihood(is_spike, conditional_intensity=None,
                       time_bin_size=1):
    '''Probability of parameters given spiking at a particular time

    Parameters
    ----------
    is_spike : array_like with values in {0, 1}, shape (n_signals,)
        Indicator of spike or no spike at current time.
    conditional_intensity : array_like, shape (n_signals,
                                               n_parameters * n_states)
        Instantaneous probability of observing a spike
    time_bin_size : float, optional

    Returns
    -------
    poisson_likelihood : array_like, shape (n_signals,
                                            n_parameters * n_states)

    '''
    probability_no_spike = np.exp(-conditional_intensity * time_bin_size)
    return ((conditional_intensity ** is_spike[:, np.newaxis]) *
            probability_no_spike)
