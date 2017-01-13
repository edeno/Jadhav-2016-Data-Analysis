import numpy as np
import pytest
from scipy.stats import multivariate_normal

from src.ripple_decoding import _mark_space_estimator


def test__mark_estimator():
    n_signals, n_marks, n_training_spikes, mark_smoothing = 20, 4, 10, 1

    test_marks = np.zeros((n_signals, n_marks)) * np.nan
    cur_spiking_neuron_ind = [0, 10]
    test_marks[cur_spiking_neuron_ind[0], :] = np.arange(1, 9, 2)
    test_marks[cur_spiking_neuron_ind[1], :] = np.arange(9, 17, 2)

    training_marks = np.zeros((n_signals, n_marks, n_training_spikes))
    training_marks[cur_spiking_neuron_ind, :, 3] = np.arange(1, 9, 2)

    mark_space_estimator = _mark_space_estimator(
        test_marks, training_marks=training_marks,
        mark_smoothing=mark_smoothing)

    true_mark1 = multivariate_normal(
        mean=np.arange(1, 9, 2),
        cov=np.identity(n_marks) * mark_smoothing).pdf(np.arange(1, 9, 2))
    true_mark2 = multivariate_normal(
        mean=np.arange(9, 17, 2),
        cov=np.identity(n_marks) * mark_smoothing).pdf(np.arange(1, 9, 2))
    true_mark3 = multivariate_normal(
        mean=np.zeros(n_marks,),
        cov=np.identity(n_marks) * mark_smoothing).pdf(np.arange(1, 9, 2))

    assert np.allclose(
        mark_space_estimator[cur_spiking_neuron_ind[0], 3], true_mark1)
    assert np.allclose(
        mark_space_estimator[cur_spiking_neuron_ind[1], 3], true_mark2)
    assert np.allclose(
        mark_space_estimator[cur_spiking_neuron_ind[0], 2], true_mark3)
