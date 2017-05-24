import numpy as np

from src.spectral.statistics import (get_normal_distribution_p_values,
                                     fisher_z_transform)


def test_get_normal_distribution_p_values():
    # approximate 97.5 percentile of the standard normal distribution
    zscore = 1.95996
    assert np.allclose(get_normal_distribution_p_values(zscore), 0.025)


def test_fisher_z_transform():
    coherency = 0.5 * np.exp(1j * np.pi / 2) * np.ones((2, 2))
    bias1, bias2 = 3, 6
    expected_difference_z = np.ones((2, 2))
    assert np.allclose(
        fisher_z_transform(
            coherency, bias1, coherency2=coherency, bias2=bias2),
        expected_difference_z)
