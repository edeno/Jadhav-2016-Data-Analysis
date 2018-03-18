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
DROP_COLUMNS = ['from_well', 'to_well', 'labeled_segments']

logger = getLogger(__name__)


def fit_position_constant(data, sampling_frequency, penalty=0):
    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    formula = 'is_spike ~ 1'
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    linear_distance = np.arange(min_distance, np.floor(max_distance) + 1)
    predict_design_matrix = np.ones((linear_distance.size, 1))

    coords = {'position': linear_distance}
    dims = ['position']

    return summarize_fit(
        results.coefficients, predict_design_matrix,
        sampling_frequency, coords, dims, design_matrix,
        is_spike.values.squeeze(), trial_id=None, AIC=results.AIC)


def fit_task(data, sampling_frequency, penalty=1E1):
    formula = 'is_spike ~ 1 + task'
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    tasks = data.task.unique()
    firing_rate = []
    multiplicative_gain = []

    for task in tasks:
        predict_data = {
            'task': task
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency))
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients))
    coords = {'task': tasks}
    dims = ['task']

    firing_rate = xr.DataArray(
        np.stack(firing_rate), dims=dims, coords=coords,
        name='firing_rate')
    multiplicative_gain = xr.DataArray(
        np.stack(multiplicative_gain), dims=dims, coords=coords,
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_turn(data, sampling_frequency, penalty=1E1):
    formula = 'is_spike ~ 1 + turn'
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    turns = data.turn.unique()
    firing_rate = []
    multiplicative_gain = []

    for turn in turns:
        predict_data = {
            'turn': turn
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency))
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients))
    coords = {'turn': turns}
    dims = ['turn']

    firing_rate = xr.DataArray(
        np.stack(firing_rate), dims=dims, coords=coords,
        name='firing_rate')
    multiplicative_gain = xr.DataArray(
        np.stack(multiplicative_gain), dims=dims, coords=coords,
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_task_by_turn(data, sampling_frequency, penalty=1E1):
    formula = 'is_spike ~ 1 + task_by_turn'
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    task_by_turns = data.task_by_turn.unique()
    firing_rate = []
    multiplicative_gain = []

    for task_by_turn in task_by_turns:
        predict_data = {
            'task_by_turn': task_by_turn
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency))
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients))
    coords = {
        'task_by_turn': task_by_turns,
    }
    dims = ['task_by_turn']

    firing_rate = xr.DataArray(
        np.stack(firing_rate), dims=dims, coords=coords,
        name='firing_rate')
    multiplicative_gain = xr.DataArray(
        np.stack(multiplicative_gain), dims=dims, coords=coords,
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_ripple_constant(ripple_locked_spikes, sampling_frequency,
                        penalty=0):
    trial_id = (ripple_locked_spikes.index
                .get_level_values('ripple_number').values)
    formula = 'is_spike ~ 1'
    response, design_matrix = dmatrices(
        formula, ripple_locked_spikes, return_type='dataframe')
    is_spike = ripple_locked_spikes.values.squeeze()

    results = fit_glm(response, design_matrix, penalty)
    time = ripple_locked_spikes.index.get_level_values('time')
    unique_time = np.unique(time.total_seconds().values)
    predict_design_matrix = np.ones((unique_time.size, 1))

    coords = {'time': unique_time}
    dims = ['time']

    return summarize_fit(
        results.coefficients, predict_design_matrix,
        sampling_frequency, coords, dims, design_matrix, is_spike,
        trial_id, results.AIC)


def fit_1D_position(data, sampling_frequency, penalty=1E1, knot_spacing=30):
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


def fit_1D_position_by_task(data, sampling_frequency, penalty=1E1,
                            knot_spacing=30):
    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    n_steps = (max_distance - min_distance) // knot_spacing
    position_knots = min_distance + np.arange(1, n_steps) * knot_spacing
    formula = (
        'is_spike ~ 1 + task_by_turn * cr(linear_distance,'
        'knots=position_knots, constraints="center")')
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    firing_rate = []
    multiplicative_gain = []

    linear_distance = np.linspace(min_distance, max_distance, 100)
    levels = data.task_by_turn.unique()

    for level in levels:
        predict_data = {
            'linear_distance': linear_distance,
            'task_by_turn': np.full_like(linear_distance, level, dtype=object)
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency))
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients))

    dims = ['task_by_turn', 'position']
    coords = {'position': linear_distance, 'task_by_turn': levels}

    firing_rate = xr.DataArray(
        np.stack(firing_rate), dims=dims, coords=coords,
        name='firing_rate')
    multiplicative_gain = xr.DataArray(
        np.stack(multiplicative_gain), dims=dims, coords=coords,
        name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_1D_position_by_speed(data, sampling_frequency, penalty=1E1,
                             knot_spacing=30):
    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    n_steps = (max_distance - min_distance) // knot_spacing
    position_knots = min_distance + np.arange(1, n_steps) * knot_spacing
    speed_knots = np.concatenate(
        (np.arange(1, 5, 2),
         np.arange(10, np.round(data.speed.max(), -1), 10)))

    formula = ('is_spike ~ 1 + te(cr(linear_distance, knots=position_knots), '
               'cr(speed, knots=speed_knots), constraints="center")')
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)
    linear_distance = np.linspace(min_distance, max_distance, 100)
    speed = np.linspace(0.0, data.speed.max(), 100)

    linear_distance, speed = np.meshgrid(linear_distance, speed)

    predict_data = {
        'linear_distance': linear_distance.ravel(),
        'speed': speed.ravel()
    }
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], predict_data)[0]

    firing_rate = get_rate(predict_design_matrix, results.coefficients,
                           sampling_frequency).reshape(linear_distance.shape).T
    multiplicative_gain = get_gain(
        predict_design_matrix, results.coefficients
        ).reshape(linear_distance.shape).T
    coords = {
        'position': np.unique(linear_distance),
        'speed': np.unique(speed)
    }
    dims = ['position', 'speed']
    firing_rate = xr.DataArray(firing_rate, dims=dims, coords=coords,
                               name='firing_rate')
    multiplicative_gain = xr.DataArray(multiplicative_gain, dims=dims,
                                       coords=coords,
                                       name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_1D_position_by_speed_and_task(data, sampling_frequency, penalty=1E1,
                                      knot_spacing=30):
    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    n_steps = (max_distance - min_distance) // knot_spacing
    position_knots = min_distance + np.arange(1, n_steps) * knot_spacing
    speed_knots = np.concatenate(
        (np.arange(1, 5, 2),
         np.arange(10, np.round(data.speed.max(), -1), 10)))
    formula = ('is_spike ~ 1 + te(cr(linear_distance, knots=position_knots), '
               'cr(speed, knots=speed_knots), constraints="center") + '
               'task_by_turn * cr(linear_distance, knots=position_knots)')
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)
    linear_distance = np.linspace(min_distance, max_distance, 100)
    speed = np.linspace(0.0, data.speed.max(), 100)

    linear_distance, speed = np.meshgrid(linear_distance, speed)
    task_by_turns = data.task_by_turn.unique()
    firing_rate = []
    multiplicative_gain = []

    for task_by_turn in task_by_turns:
        predict_data = {
            'linear_distance': linear_distance.ravel(),
            'speed': speed.ravel(),
            'task_by_turn': np.full_like(speed.ravel(), task_by_turn, dtype=object)
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]
        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency).reshape(linear_distance.shape).T)
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients
                     ).reshape(linear_distance.shape).T)
    coords = {
        'position': np.unique(linear_distance),
        'speed': np.unique(speed),
        'task_by_turn': task_by_turns,
    }
    dims = ['task_by_turn', 'position', 'speed']
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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_1D_position_by_speed_by_task(data, sampling_frequency, penalty=1E1,
                                     knot_spacing=30):
    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    n_steps = (max_distance - min_distance) // knot_spacing
    position_knots = min_distance + np.arange(1, n_steps) * knot_spacing
    speed_knots = np.concatenate(
        (np.arange(1, 5, 2),
         np.arange(10, np.round(data.speed.max(), -1), 10)))

    formula = ('is_spike ~ 1 + task_by_turn * '
               'te(cr(linear_distance, knots=position_knots), '
               'cr(speed, knots=speed_knots), constraints="center")')
    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)
    linear_distance = np.linspace(min_distance, max_distance, 100)
    speed = np.linspace(0.0, data.speed.max(), 100)

    linear_distance, speed = np.meshgrid(linear_distance, speed)
    task_by_turns = data.task_by_turn.unique()
    firing_rate = []
    multiplicative_gain = []

    for task_by_turn in task_by_turns:
        predict_data = {
            'linear_distance': linear_distance.ravel(),
            'speed': speed.ravel(),
            'task_by_turn': np.full_like(speed.ravel(), task_by_turn,
                                         dtype=object)
        }
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]

        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency).reshape(linear_distance.shape).T)
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients
                     ).reshape(linear_distance.shape).T)
    coords = {
        'position': np.unique(linear_distance),
        'speed': np.unique(speed),
        'task_by_turn': task_by_turns,
    }
    dims = ['task_by_turn', 'position', 'speed']
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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_2D_position(data, neuron_key, animals, sampling_frequency,
                    penalty=1E1):

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

    firing_rate = get_rate(predict_design_matrix, results.coefficients,
                           sampling_frequency).reshape(x.shape).T
    multiplicative_gain = get_gain(
        predict_design_matrix, results.coefficients).reshape(x.shape).T
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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_2D_position_by_task(data, neuron_key, animals, sampling_frequency,
                            penalty=1E1):
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

        rate = get_rate(predict_design_matrix, results.coefficients,
                        sampling_frequency)
        firing_rate.append(rate.reshape(x.shape).T)

        gain = get_gain(predict_design_matrix, results.coefficients)
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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_2D_position_by_speed(data, neuron_key, animals, sampling_frequency,
                             penalty=1E1):

    x_knots, y_knots = get_position_knots(neuron_key[:-2], animals)
    speed_knots = np.concatenate(
        (np.arange(1, 5, 2),
         np.arange(10, np.round(data.speed.max(), -1), 10)))

    formula = ('is_spike ~ 1 + te(cr(x_position, knots=x_knots), '
               'cr(y_position, knots=y_knots), cr(speed, knots=speed_knots), '
               'constraints="center")')

    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

    x = np.linspace(data.x_position.min(), data.x_position.max(), 50)
    y = np.linspace(data.y_position.min(), data.y_position.max(), 50)
    x, y = np.meshgrid(x, y)

    firing_rate = []
    multiplicative_gain = []

    for speed in _SPEEDS:
        predict_data = {'x_position': x.ravel(), 'y_position': y.ravel(),
                        'speed': np.ones_like(x.ravel()) * speed}
        predict_design_matrix = build_design_matrices(
            [design_matrix.design_info], predict_data)[0]

        rate = get_rate(predict_design_matrix, results.coefficients,
                        sampling_frequency)
        firing_rate.append(rate.reshape(x.shape).T)

        gain = get_gain(predict_design_matrix, results.coefficients)
        multiplicative_gain.append(gain.reshape(x.shape).T)

    coords = {
        'speed': _SPEEDS,
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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_2D_position_by_speed_and_task(data, neuron_key, animals,
                                      sampling_frequency, penalty=1E1):
    x_knots, y_knots = get_position_knots(neuron_key[:-2], animals)
    speed_knots = np.concatenate(
        (np.arange(1, 5, 2),
         np.arange(10, np.round(data.speed.max(), -1), 10)))

    formula = (
        'is_spike ~ 1 + te(cr(x_position, knots=x_knots), '
        'cr(y_position, knots=y_knots), cr(speed, knots=speed_knots), '
        'constraints="center") + task * te(cr(x_position, knots=x_knots), '
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
        temp_firing_rate = []
        temp_multiplicative_gain = []
        for speed in _SPEEDS:
            predict_data = {'x_position': x.ravel(), 'y_position': y.ravel(),
                            'speed': np.ones_like(x.ravel()) * speed,
                            'task': np.full_like(x.ravel(), task, dtype=object)
                            }
            predict_design_matrix = build_design_matrices(
                [design_matrix.design_info], predict_data)[0]

            rate = get_rate(predict_design_matrix, results.coefficients,
                            sampling_frequency)
            temp_firing_rate.append(rate.reshape(x.shape).T)

            gain = get_gain(predict_design_matrix, results.coefficients)
            temp_multiplicative_gain.append(gain.reshape(x.shape).T)
        firing_rate.append(temp_firing_rate)
        multiplicative_gain.append(temp_multiplicative_gain)

    coords = {
        'task': tasks,
        'speed': _SPEEDS,
        'x_position': np.unique(x),
        'y_position': np.unique(y),
    }
    dims = ['task', 'speed', 'x_position', 'y_position']
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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_ripple_over_time(ripple_locked_spikes, sampling_frequency,
                         penalty=1E1, knot_spacing=0.050):
    time = ripple_locked_spikes.index.get_level_values('time')
    unique_time = np.unique(time.total_seconds().values)
    trial_id = (ripple_locked_spikes.index
                .get_level_values('ripple_number').values)

    n_steps = (unique_time[-1] - unique_time[0]) // knot_spacing
    time_knots = unique_time[0] + np.arange(1, n_steps) * knot_spacing
    formula = '1 + cr(time, knots=time_knots, constraints="center")'
    design_matrix = dmatrix(
        formula, dict(time=time.total_seconds().values),
        return_type='dataframe')
    is_spike = ripple_locked_spikes.values.squeeze()

    results = fit_glm(is_spike, design_matrix, penalty)
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], dict(time=unique_time))[0]

    coords = {'time': unique_time}
    dims = ['time']

    return summarize_fit(
        results.coefficients, predict_design_matrix,
        sampling_frequency, coords, dims, design_matrix, is_spike,
        trial_id, results.AIC)


def fit_replay(ripple_locked_spikes, sampling_frequency,
               replay_info, covariate, penalty=1E1, knot_spacing=0.050):

    time = ripple_locked_spikes.unstack(level=0).index.total_seconds().values

    trial_id = (ripple_locked_spikes.index
                .get_level_values('ripple_number').values)
    n_steps = (time[-1] - time[0]) // knot_spacing
    time_knots = time[0] + np.arange(1, n_steps) * knot_spacing

    data = (pd.merge(ripple_locked_spikes.reset_index(), replay_info,
                     on='ripple_number')
            .assign(time=lambda df: df.time.dt.total_seconds()))
    formula = (f'is_spike ~ {covariate} * '
               'cr(time, knots=time_knots, constraints="center")')

    is_spike, design_matrix = dmatrices(formula, data, return_type='dataframe')
    results = fit_glm(is_spike, design_matrix, penalty)

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
        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency))
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients))

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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
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
        firing_rate.append(
            get_rate(predict_design_matrix, results.coefficients,
                     sampling_frequency))
        multiplicative_gain.append(
            get_gain(predict_design_matrix, results.coefficients))

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

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike.squeeze()
                      ).ks_statistic(), name='ks_statistic')
    AIC = xr.DataArray(results.AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain, baseline_firing_rate,
                     ks_statistic, AIC))


def fit_hippocampal_theta(data, sampling_frequency, penalty=1E1):
    formula = ('is_spike ~ 1 + np.cos(instantaneous_phase)'
               ' + np.sin(instantaneous_phase)')
    is_spike, design_matrix = dmatrices(formula, data)
    results = fit_glm(is_spike, design_matrix, penalty)
    predict_data = {'instantaneous_phase': np.linspace(-np.pi, np.pi)}
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], predict_data)[0]

    firing_rate = xr.DataArray(
        get_rate(predict_design_matrix, results.coefficients,
                 sampling_frequency), dims=['instantaneous_phase'],
        coords=predict_data, name='firing_rate')
    phase_vector = results.coefficients[1] + 1j * results.coefficients[2]
    preferred_phase = xr.DataArray(np.angle(phase_vector),
                                   name='preferred_phase')
    modulation = xr.DataArray(np.abs(phase_vector), name='modulation')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')
    aic = xr.DataArray(results.AIC, name='AIC')
    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')

    return xr.merge((firing_rate, preferred_phase, modulation,
                     baseline_firing_rate, aic, ks_statistic))


def fit_hippocampal_theta_by_1D_position(data, sampling_frequency,
                                         knot_spacing=30, penalty=1E1):

    min_distance, max_distance = (data.linear_distance.min(),
                                  data.linear_distance.max())
    n_steps = (max_distance - min_distance) // knot_spacing
    position_knots = min_distance + np.arange(1, n_steps) * knot_spacing
    phase_knots = np.arange(-np.pi, np.pi, np.pi / 4)[1:]
    formula = ('is_spike ~ 1 + te(cr(linear_distance, knots=position_knots), '
               'cr(instantaneous_phase, knots=phase_knots), '
               'constraints="center")')
    is_spike, design_matrix = dmatrices(formula, data)
    results = fit_glm(is_spike, design_matrix, penalty)

    instantaneous_phase = np.linspace(-np.pi, np.pi)
    linear_distance = np.arange(min_distance, np.floor(max_distance) + 1)
    linear_distance, instantaneous_phase = np.meshgrid(
        linear_distance, instantaneous_phase)
    predict_data = {
        'instantaneous_phase': instantaneous_phase.ravel(),
        'linear_distance': linear_distance.ravel(),
    }
    predict_design_matrix = build_design_matrices(
        [design_matrix.design_info], predict_data)[0]
    firing_rate = get_rate(
        predict_design_matrix, results.coefficients,
        sampling_frequency).reshape(instantaneous_phase.shape).T
    multiplicative_gain = get_gain(
        predict_design_matrix, results.coefficients
        ).reshape(instantaneous_phase.shape).T
    coords = {
        'phase': np.unique(instantaneous_phase),
        'position': np.unique(linear_distance),
    }
    dims = ['position', 'phase']
    firing_rate = xr.DataArray(firing_rate, dims=dims, coords=coords,
                               name='firing_rate')
    multiplicative_gain = xr.DataArray(multiplicative_gain, dims=dims,
                                       coords=coords,
                                       name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(
        np.exp(results.coefficients[0]) * sampling_frequency,
        name='baseline_firing_rate')

    conditional_intensity = get_rate(design_matrix, results.coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity,
                      is_spike.squeeze()).ks_statistic(),
        name='ks_statistic')
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


def summarize_fit(coefficients, predict_design_matrix,
                  sampling_frequency, coords, dims, design_matrix, is_spike,
                  trial_id=None, AIC=np.nan):

    firing_rate = xr.DataArray(
        get_rate(predict_design_matrix, coefficients,
                 sampling_frequency), dims=dims,
        coords=coords, name='firing_rate')
    try:
        multiplicative_gain = xr.DataArray(
            get_gain(predict_design_matrix, coefficients),
            dims=dims, coords=coords,
            name='multiplicative_gain')
    except ValueError:
        multiplicative_gain = xr.DataArray([],
                                           dims=dims, coords=coords,
                                           name='multiplicative_gain')
    baseline_firing_rate = xr.DataArray(np.squeeze(np.exp(
        np.atleast_1d(coefficients)[0]) * sampling_frequency),
        name='baseline_firing_rate')
    conditional_intensity = get_rate(design_matrix, coefficients)
    ks_statistic = xr.DataArray(
        TimeRescaling(conditional_intensity, is_spike).ks_statistic(),
        name='ks_statistic')
    AIC = xr.DataArray(AIC, name='AIC')

    return xr.merge((firing_rate, multiplicative_gain,
                     baseline_firing_rate, ks_statistic,
                     AIC))


def cluster(data, maxgap=15):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

    >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

    >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
        https://stackoverflow.com/questions/14783947/grouping-clustering-numbers-in-python
    '''
    data = np.sort(data.copy())
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

    knots = [insert_points([np.mean(x) for x in cluster(dim)],
                           min_diff=knot_spacing)
             for dim in coordinates.T]
    return tuple(knots)


def lag(df, trial=None, n_lags=1, fillna_value=0.0):
    df = pd.Series(df) if not isinstance(df, pd.Series) else df
    df = df.groupby(trial) if trial is not None else df
    return pd.concat([df.shift(lag).fillna(fillna_value)
                      for lag in np.arange(1, n_lags + 1)], axis=1)


def get_rate(design_matrix, coefficients, sampling_frequency=1):
    return np.squeeze(
        np.exp(np.dot(design_matrix, coefficients)) * sampling_frequency)


def get_gain(design_matrix, coefficients):
    try:
        return np.squeeze(
            np.exp(np.dot(design_matrix[:, 1:], coefficients[1:])))
    except IndexError:
        return np.ones((design_matrix.shape[0],))
