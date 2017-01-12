import numpy as np
import pandas as pd
import pytest

from src.ripple_detection import (_find_containing_interval,
                                  _get_series_start_end_times,
                                  _extend_segment,
                                  segment_boolean_series)


@pytest.mark.parametrize("series, expected_segments", [
    (pd.Series([False, False, True, True, False]),
     (np.array([2]), np.array([3]))),
    (pd.Series([False, False, True, True, False, True, False]),
     (np.array([2, 5]), np.array([3, 5]))),
    (pd.Series([True, True, False, False, False]),
     (np.array([0]), np.array([1]))),
    (pd.Series([False, False, True, True, True]),
     (np.array([2]), np.array([4]))),
    (pd.Series([True, False, True, True, False]),
     (np.array([0, 2]), np.array([0, 3]))),
])
def test_get_series_start_end_times(series, expected_segments):
    tup = _get_series_start_end_times(series)
    try:
        assert np.all(tup[0] == expected_segments[0]) & np.all(
            tup[1] == expected_segments[1])
    except IndexError:
        assert tup == expected_segments


@pytest.mark.parametrize("series, expected_segments", [
    (pd.Series([False, True, True, True, False],
               index=np.linspace(0, 0.020, 5)), [(0.005, 0.015)]),
    (pd.Series([False, False, True, True, False, True, False],
               index=np.linspace(0, 0.030, 7)), []),
    (pd.Series([True, True, False, False, False],
               index=np.linspace(0, 0.020, 5)), []),
    (pd.Series([False, True, True, True, True],
               index=np.linspace(0, 0.020, 5)), [(0.005, 0.020)]),
    (pd.Series([True, True, True, True, False],
               index=np.linspace(0, 0.020, 5)), [(0.000, 0.015)]),
    (pd.Series([True, True, True, True, False, True, True, True],
               index=np.linspace(0, 0.035, 8)),
        [(0.000, 0.015), (0.025, 0.035)]),
])
def test_segment_boolean_series(series, expected_segments):
    assert np.all(
        [(np.allclose(expected_start, test_start)) &
         (np.allclose(expected_end, test_end))
         for (test_start, test_end), (expected_start, expected_end)
         in zip(segment_boolean_series(series), expected_segments)])


@pytest.mark.parametrize(
    "interval_candidates, target_interval, expected_interval", [
        ([(1, 2), (5, 7)], (6, 7), (5, 7)),
        ([(1, 2), (5, 7)], (1, 2), (1, 2)),
        ([(1, 2), (5, 7), (20, 30)], (5, 6), (5, 7)),
        ([(1, 2), (5, 7), (20, 30)], (24, 26), (20, 30)),
    ])
def test_find_containing_interval(interval_candidates, target_interval,
                                  expected_interval):
    test_interval = _find_containing_interval(
        interval_candidates, target_interval)
    assert np.all(test_interval == expected_interval)


@pytest.mark.parametrize(
    "interval_candidates, target_intervals, expected_intervals", [
        ([(1, 2), (5, 7)], [(6, 7)], [(5, 7)]),
        ([(1, 2), (5, 7)], [(1, 2)], [(1, 2)]),
        ([(1, 2), (5, 7), (20, 30)], [(5, 6)], [(5, 7)]),
        ([(1, 2), (5, 7), (20, 30)], [(24, 26), (6, 7)], [(5, 7),
                                                          (20, 30)]),
        ([(1, 2), (5, 7), (20, 30)], [(24, 26), (27, 28)], [(20, 30)]),
    ])
def test__extend_segment(interval_candidates, target_intervals,
                         expected_intervals):
    test_intervals = _extend_segment(
        target_intervals, interval_candidates)
    assert np.all(test_intervals == expected_intervals)
