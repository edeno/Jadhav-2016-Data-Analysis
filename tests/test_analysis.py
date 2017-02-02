import numpy as np
import pandas as pd

from src.analysis import merge_symmetric_key_pairs


def test_merge_symmetric_key_pairs():
    test_dict = {('a', 'a'): pd.Index([1, 2, 3]),
                 ('a', 'b'): pd.Index([4, 5, 6]),
                 ('b', 'a'): pd.Index([7, 8, 9]),
                 ('b', 'c'): pd.Index([10, 11, 12])}
    merged_dict = merge_symmetric_key_pairs(test_dict)
    expected_dict = {('a', 'a'): pd.Index([1, 2, 3]),
                     ('a', 'b'): pd.Index([4, 5, 6, 7, 8, 9]),
                     ('b', 'c'): pd.Index([10, 11, 12])}
    assert expected_dict.keys() == merged_dict.keys()
    assert all(
        [np.all(expected_dict[expected_key] == merged_dict[expected_key])
         for expected_key in expected_dict])
