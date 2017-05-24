import numpy as np

from src.spectral.statistics import (get_normal_distribution_p_values)


def test_get_normal_distribution_p_values():
    # approximate 97.5 percentile of the standard normal distribution
    zscore = 1.95996
    assert np.allclose(get_normal_distribution_p_values(zscore), 0.025)
