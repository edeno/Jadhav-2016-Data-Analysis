from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from patsy import build_design_matrices, dmatrices, dmatrix
from statsmodels.api import families

from loren_frank_data_processing import (get_spike_indicator_dataframe,
                                         reshape_to_segments)
from loren_frank_data_processing.core import get_data_structure
from regularized_glm import penalized_IRLS
from time_rescale import TimeRescaling

_SPEEDS = [1, 2, 3, 4, 10, 20, 30, 40]
DROP_COLUMNS = ['from_well', 'to_well', 'labeled_segments', 'is_correct',
                'task']
SPEED_KNOTS = [1, 3, 10, 30]

logger = getLogger(__name__)


def fit_constant(neuron_key, animals, penalty=None):
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    formula = 'is_spike ~ 1'
    is_spike, design_matrix = dmatrices(
        formula, spikes, return_type='dataframe')
    model_coefficients, AIC, _ = fit_glm(is_spike, design_matrix, penalty)


def fit_ripple_constant(neuron_key, animals, sampling_frequency, ripple_times,
                        window_offset=(-0.500, 0.500), penalty=None):
    logger.info(f'Fitting ripple constant model for {neuron_key}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    ripple_locked_spikes = reshape_to_segments(
        spikes, ripple_times, window_offset, sampling_frequency)
    trial_id = (ripple_locked_spikes.index
                .get_level_values('ripple_number').values)
    formula = 'is_spike ~ 1'
    response, design_matrix = dmatrices(
        formula, ripple_locked_spikes, return_type='dataframe')
    is_spike = ripple_locked_spikes.values.squeeze()

    results = fit_glm(response, design_matrix, penalty)
    time = ripple_locked_spikes.index.get_level_values('time')
    unique_time = time.unique().total_seconds().values
    predict_design_matrix = np.ones((unique_time.size, 1))

    coords = {'time': unique_time}
    dims = ['time']

    return summarize_fit(
        results.coefficients, predict_design_matrix,
        sampling_frequency, coords, dims, design_matrix, is_spike,
        trial_id, results.AIC)


def fit_1D_position(neuron_key, animals, sampling_frequency, position_info,
                    penalty=1E-4, knot_spacing=30):
    logger.info(f'Fitting 1D position model for {neuron_key}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    data = (position_info.join(spikes)
            .drop(DROP_COLUMNS, axis=1)
            .dropna())
    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    n_steps = (max_distance - min_distance) // knot_spacing
    position_knots = min_distance + np.arange(1, n_steps) * knot_spacing
    formula = ('is_spike ~ 1 + cr(linear_distance, knots=position_knots,'
               ' constraints="center")')
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    predict_data = {
        'linear_distance': np.arange(min_distance, np.floor(max_distance) + 1)}
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], predict_data)[0]

    coords = {'position': predict_data['linear_distance']}
    dims = ['position']

    return summarize_fit(
        results.coefficients, predict_design_matrix,
        sampling_frequency, coords, dims, design_matrix,
        is_spike.values.squeeze(), trial_id=None, AIC=results.AIC)


def fit_1D_position_by_task(
        neuron_key, animals, sampling_frequency, position_info,
        penalty=1E1, knot_spacing=30):
    logger.info(f'Fitting 1D position by task model for {neuron_key}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    data = (position_info.join(spikes)
            .drop(DROP_COLUMNS, axis=1)
            .dropna().query('speed > 4'))
    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    n_steps = (max_distance - min_distance) // knot_spacing
    position_knots = min_distance + np.arange(1, n_steps) * knot_spacing
    formula = (
        'is_spike ~ 1 + task * cr(linear_distance,'
        'knots=position_knots, constraints="center")')
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    firing_rate = []
    multiplicative_gain = []

    linear_distance = np.linspace(min_distance, max_distance, 100)
    levels = ['Inbound', 'Outbound']

    for level in levels:
        predict_data = {
            'linear_distance': linear_distance,
            'task': np.full_like(linear_distance, level, dtype=object)
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(np.squeeze(
            np.exp(predict_design_matrix.dot(results.coefficients)) *
            sampling_frequency))
        multiplicative_gain.append(np.squeeze(
            np.exp(predict_design_matrix[:, 1:].dot(results.coefficients[1:])))
        )

    dims = ['task', 'position']
    coords = {'position': linear_distance, 'task': levels}

    firing_rate = xr.DataArray(
        np.stack(firing_rate), dims=dims, coords=coords,
        name='firing_rate')
    multiplicative_gain = xr.DataArray(
        np.stack(multiplicative_gain), dims=dims, coords=coords,
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = np.exp(design_matrix @ results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_2D_position(neuron_key, animals, sampling_frequency, position_info,
                    penalty=1E-4):
    logger.info(f'Fitting 2D position model for {neuron_key}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    data = (position_info.join(spikes)
            .drop(DROP_COLUMNS, axis=1)
            .dropna())
    x_knots, y_knots = get_position_knots(neuron_key[:-2], animals)
    formula = ('is_spike ~ 1 + te(cr(x_position, knots=x_knots), '
               'cr(y_position, knots=y_knots), constraints="center")')

    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    x = np.linspace(data.x_position.min(), data.x_position.max(), 50)
    y = np.linspace(data.y_position.min(), data.y_position.max(), 50)
    x, y = np.meshgrid(x, y)

    predict_data = {'x_position': x.ravel(), 'y_position': y.ravel()}
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], predict_data)[0]

    firing_rate = (np.exp(np.dot(predict_design_matrix, model_coefficients))
                   * sampling_frequency).reshape(x.shape)
    multiplicative_gain = np.exp(
        predict_design_matrix[:, 1:] @ results.coefficients[1:]
    ).reshape(x.shape).T
    coords = {
        'x_position': np.unique(x),
        'y_position': np.unique(y),
    }
    dims = ['x_position', 'y_position']
    firing_rate = xr.DataArray(firing_rate, dims=dims, coords=coords,
                               name='firing_rate')
    multiplicative_gain = xr.DataArray(multiplicative_gain, dims=dims,
                                       coords=coords,
                                       name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = np.exp(design_matrix @ results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_2D_position_by_task(neuron_key, animals, sampling_frequency,
                            position_info, penalty=1E1):
    logger.info(f'Fitting 2D position model by task for {neuron_key}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    data = (position_info.join(spikes)
            .drop(DROP_COLUMNS, axis=1)
            .dropna())
    x_knots, y_knots = get_position_knots(neuron_key[:-2], animals)
    formula = ('is_spike ~ 1 + task * te(cr(x_position, knots=x_knots), '
               'cr(y_position, knots=y_knots), constraints="center")')

    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    x = np.linspace(data.x_position.min(), data.x_position.max(), 50)
    y = np.linspace(data.y_position.min(), data.y_position.max(), 50)
    x, y = np.meshgrid(x, y)

    tasks = ['Inbound', 'Outbound']
    firing_rate = []
    multiplicative_gain = []

    for task in tasks:
        predict_data = {'x_position': x.ravel(), 'y_position': y.ravel(),
                        'task': np.full_like(x.ravel(), task, dtype=object)}
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]

        rate = (np.exp(predict_design_matrix @ results.coefficients)
                * sampling_frequency)
        firing_rate.append(rate.reshape(x.shape).T)

        gain = np.exp(predict_design_matrix[:, 1:] @ results.coefficients[1:])
        multiplicative_gain.append(gain.reshape(x.shape).T)

    coords = {
        'task': tasks,
        'x_position': np.unique(x),
        'y_position': np.unique(y),
    }
    dims = ['task', 'x_position', 'y_position']
    firing_rate = xr.DataArray(np.stack(firing_rate),
                               dims=dims,
                               coords=coords,
                               name='firing_rate')
    multiplicative_gain = xr.DataArray(np.stack(multiplicative_gain),
                                       dims=dims,
                                       coords=coords,
                                       name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = np.exp(design_matrix @ results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))

def fit_2D_position_by_speed(neuron_key, animals, sampling_frequency,
                              position_info, speeds=_SPEEDS, penalty=1E1):
    logger.info(f'Fitting 2D position and speed model for {neuron_key}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    data = (position_info.join(spikes)
            .drop(DROP_COLUMNS, axis=1)
            .dropna())
    x_knots, y_knots = get_position_knots(neuron_key[:-2], animals)
    formula = ('is_spike ~ 1 + te(cr(x_position, knots=x_knots), '
               'cr(y_position, knots=y_knots), cr(speed, knots=SPEED_KNOTS), '
               'constraints="center")')

    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    x = np.linspace(data.x_position.min(), data.x_position.max(), 50)
    y = np.linspace(data.y_position.min(), data.y_position.max(), 50)
    x, y = np.meshgrid(x, y)

    firing_rate = []
    multiplicative_gain = []

    for speed in speeds:
        predict_data = {'x_position': x.ravel(), 'y_position': y.ravel(),
                        'speed': np.ones_like(x.ravel()) * speed}
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]

        rate = (np.exp(predict_design_matrix @ results.coefficients)
                * sampling_frequency)
        firing_rate.append(rate.reshape(x.shape).T)

        gain = np.exp(predict_design_matrix[:, 1:] @ results.coefficients[1:])
        multiplicative_gain.append(gain.reshape(x.shape).T)

    coords = {
        'speed': speeds,
        'x_position': np.unique(x),
        'y_position': np.unique(y),
    }
    dims = ['speed', 'x_position', 'y_position']
    firing_rate = xr.DataArray(np.stack(firing_rate),
                               dims=dims,
                               coords=coords,
                               name='firing_rate')
    multiplicative_gain = xr.DataArray(np.stack(multiplicative_gain),
                                       dims=dims,
                                       coords=coords,
                                       name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = np.exp(design_matrix @ results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_ripple_over_time(neuron_key, animals, sampling_frequency, ripple_times,
                         window_offset=(-0.500, 0.500),
                         penalty=1E-4, knot_spacing=0.050):
    logger.info(f'Fitting ripple spline model for {neuron_key}')
    spikes = get_spike_indicator_dataframe(neuron_key, animals)
    ripple_locked_spikes = reshape_to_segments(
        spikes, ripple_times, window_offset, sampling_frequency)
    time = ripple_locked_spikes.index.get_level_values('time')
    trial_id = (ripple_locked_spikes.index
                .get_level_values('ripple_number').values)
    n_steps = np.diff(window_offset) // knot_spacing
    time_knots = window_offset[0] + np.arange(1, n_steps) * knot_spacing
    formula = '1 + cr(time, knots=time_knots, constraints="center")'
    design_matrix = dmatrix(
        formula, dict(time=time.total_seconds().values),
        return_type='dataframe')
    is_spike = ripple_locked_spikes.values.squeeze()

    results = fit_glm(is_spike, design_matrix, penalty)
    unique_time = time.unique().total_seconds().values
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], dict(time=unique_time))[0]

    coords = {'time': unique_time}
    dims = ['time']

    return summarize_fit(
        results.coefficients, predict_design_matrix,
        sampling_frequency, coords, dims, design_matrix, is_spike,
        trial_id, results.AIC)


def fit_replay(neuron_key, animals, sampling_frequency,
               replay_info, covariate, window_offset=(-0.500, 0.500),
               penalty=1E-4, knot_spacing=0.050):
    logger.info(f'Fitting replay model for {neuron_key} - {covariate}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    ripple_times = (replay_info.set_index('ripple_number')
                    .loc[:, ['start_time', 'end_time']])
    ripple_locked_spikes = reshape_to_segments(
        spikes, ripple_times, window_offset, sampling_frequency)
    trial_id = (ripple_locked_spikes.index
                .get_level_values('ripple_number').values)
    n_steps = np.diff(window_offset) // knot_spacing
    time_knots = window_offset[0] + np.arange(1, n_steps) * knot_spacing

    data = (pd.merge(ripple_locked_spikes.reset_index(), replay_info,
                     on='ripple_number')
            .assign(time=lambda df: df.time.dt.total_seconds()))
    formula = ('is_spike ~ {covariate} * '
               'cr(time, knots=time_knots, constraints="center")').format(
        covariate=covariate)

    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)
    time = ripple_locked_spikes.unstack(level=0).index.total_seconds().values
    levels = data[covariate].unique()
    firing_rate = []
    multiplicative_gain = []

    for level in levels:
        predict_data = {
            'time': time,
            covariate: np.full_like(time, level, dtype=object)
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(np.squeeze(
            np.exp(predict_design_matrix.dot(results.coefficients))) *
            sampling_frequency)
        multiplicative_gain.append(np.squeeze(
            np.exp(predict_design_matrix[:, 1:].dot(results.coefficients[1:])))
        )

    dims = [covariate, 'time']
    coords = {'time': time, covariate: levels}

    firing_rate = xr.DataArray(
        np.stack(firing_rate), dims=dims, coords=coords,
        name='firing_rate')
    multiplicative_gain = xr.DataArray(
        np.stack(multiplicative_gain), dims=dims, coords=coords,
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = np.exp(design_matrix @ results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_replay_no_interaction(neuron_key, animals, sampling_frequency,
                              replay_info, covariate, window_offset=(
                                  -0.500, 0.500),
                              penalty=1E1, knot_spacing=0.050):
    logger.info(f'Fitting replay model for {neuron_key} - {covariate}')
    spikes = get_spike_indicator_dataframe(
        neuron_key, animals).rename('is_spike')
    ripple_times = (replay_info.set_index('ripple_number')
                    .loc[:, ['start_time', 'end_time']])
    ripple_locked_spikes = reshape_to_segments(
        spikes, ripple_times, window_offset, sampling_frequency)
    trial_id = (ripple_locked_spikes.index
                .get_level_values('ripple_number').values)
    n_steps = np.diff(window_offset) // knot_spacing
    time_knots = window_offset[0] + np.arange(1, n_steps) * knot_spacing

    data = (pd.merge(ripple_locked_spikes.reset_index(), replay_info,
                     on='ripple_number')
            .assign(time=lambda df: df.time.dt.total_seconds()))
    formula = ('is_spike ~ {covariate} + '
               'cr(time, knots=time_knots, constraints="center")').format(
        covariate=covariate)

    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)
    time = ripple_locked_spikes.unstack(level=0).index.total_seconds().values
    levels = data[covariate].unique()
    firing_rate = []
    multiplicative_gain = []

    for level in levels:
        predict_data = {
            'time': time,
            covariate: np.full_like(time, level, dtype=object)
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(np.squeeze(
            np.exp(predict_design_matrix.dot(results.coefficients)) *
            sampling_frequency))
        multiplicative_gain.append(np.squeeze(
            np.exp(predict_design_matrix[:, 1:].dot(results.coefficients[1:])))
        )

    dims = [covariate, 'time']
    coords = {'time': time, covariate: levels}

    firing_rate = xr.DataArray(
        np.stack(firing_rate), dims=dims, coords=coords,
        name='firing_rate')
    multiplicative_gain = xr.DataArray(
        np.stack(multiplicative_gain), dims=dims, coords=coords,
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = np.exp(design_matrix @ results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_glm(response, design_matrix, penalty=None):
    if penalty is not None:
        penalty = np.ones((design_matrix.shape[1],)) * penalty
        penalty[0] = 0.0  # don't penalize the intercept
    else:
        penalty = np.finfo(np.float).eps
    return penalized_IRLS(
        np.array(design_matrix), np.array(response).squeeze(),
        family=families.Poisson(), penalty=penalty)


def summarize_fit(model_coefficients, predict_design_matrix,
                  sampling_frequency, coords, dims, design_matrix, is_spike,
                  trial_id=None, AIC=np.nan):
    if np.allclose(model_coefficients, 0):
        model_coefficients *= np.nan
    firing_rate = xr.DataArray(
        np.squeeze(np.exp(predict_design_matrix @ model_coefficients) *
                   sampling_frequency), dims=dims,
        coords=coords, name='firing_rate')
    try:
        multiplicative_gain = xr.DataArray(np.squeeze(np.exp(
            predict_design_matrix[:, 1:] @ model_coefficients[1:])),
            dims=dims, coords=coords,
            name='multiplicative_gain')
    except ValueError:
        multiplicative_gain = xr.DataArray([],
                                           dims=dims, coords=coords,
                                           name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(np.exp(
        model_coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')
    conditional_intensity = np.exp(design_matrix @ model_coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain,
                     baseline_firing_rate, ks_statistic,
                     AIC))


def cluster(data, maxgap=5):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

    >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

    >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
        https://stackoverflow.com/questions/14783947/grouping-clustering-numbers-in-python
    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def insert_points(data, min_diff=30):
    new_points = []
    for i, d in enumerate(np.diff(data)):
        if d > min_diff:
            n = np.ceil(d / min_diff).astype(int)
            new_diff = d / n
            new_points.append(data[i] + np.arange(1, n) * new_diff)
    return np.sort(np.concatenate((data, *new_points)))


def get_position_knots(epoch_key, animals, knot_spacing=30):
    animal, day, epoch = epoch_key
    task_file = get_data_structure(animals[animal], day, 'task', 'task')
    linearcoord = task_file[epoch - 1]['linearcoord'][0, 0].squeeze()

    coordinates = np.concatenate([arm[:, :, 0] for arm in linearcoord])
    coordinates = np.unique(coordinates, axis=0)

    knots = [insert_points(np.mean(cluster(dim), axis=1),
                           min_diff=knot_spacing)
             for dim in coordinates.T]
    return tuple(knots)


def lag(df, trial=None, n_lags=1, fillna_value=0.0):
    df = df.groupby(trial) if trial is not None else df
    return pd.concat([df.shift(lag).fillna(fillna_value)
                      for lag in np.arange(1, n_lags + 1)], axis=1)
