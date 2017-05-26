import numpy as np
from pytest import mark

from src.spectral.minimum_phase_decomposition import (
    _check_convergence, _conjugate_transpose)


def test__check_convergence():
    tolerance = 1e-8
    n_time_points = 5
    minimum_phase_factor = np.zeros((n_time_points, 4, 3))
    old_minimum_phase_factor = np.zeros((n_time_points, 4, 3))
    minimum_phase_factor[0, :, :] = 1e-9
    minimum_phase_factor[1, :, :] = 1e-7
    minimum_phase_factor[3, :] = 1
    minimum_phase_factor[4, :3, 1:2] = 1e-7

    expected_is_converged = np.array([True, False, True, False, False])

    is_converged = _check_convergence(
        minimum_phase_factor, old_minimum_phase_factor, tolerance)

    assert np.all(is_converged == expected_is_converged)


def test__conjugate_transpose():
    test_array = np.zeros((2, 2, 4), dtype=np.complex)
    test_array[1, ...] = [[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
                          [1 - 2j, 3 - 4j, 5 - 6j, 7 - 8j]]
    expected_array = np.zeros((2, 4, 2), dtype=np.complex)
    expected_array[1, ...] = test_array[1, ...].conj().transpose()
    assert np.allclose(_conjugate_transpose(test_array), expected_array)
