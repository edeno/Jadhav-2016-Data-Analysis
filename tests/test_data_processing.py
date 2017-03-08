''' Testing for the data processing module. '''
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (find_closest_ind,
                                 get_data_filename, get_epochs,
                                 merge_symmetric_key_pairs)


@pytest.mark.parametrize('day, expected_name', [
    (2, '/Raw-Data/test_dir/Testdummy02.mat'),
    (11, '/Raw-Data/test_dir/Testdummy11.mat'),
])
def test_data_file_name_returns_correct_file(day, expected_name):
    Animal = namedtuple('Animal', {'directory', 'short_name'})
    animal = Animal(directory='test_dir', short_name='Test')
    file_type = 'dummy'

    file_name = get_data_filename(animal, day, file_type)
    assert expected_name in file_name


@pytest.mark.parametrize('search_array, target, expected_index', [
    (np.arange(50, 150), 66, 16),
    (np.arange(50, 150), 45, 0),
    (np.arange(50, 150), 200, 99),
    (np.arange(50, 150), 66.4, 16),
    (np.arange(50, 150), 66.7, 17),
    (np.arange(50, 150), [55, 65, 137], [5, 15, 87]),
])
def test_find_closest_ind(search_array, target, expected_index):
    assert np.all(find_closest_ind(
        search_array, target) == expected_index)

# Create a fake 'tasks' data set to test
mock_data_struct = np.zeros(5, dtype={'names': ['type', 'environment'],
                                      'formats': ['O', 'O']})
mock_data_struct[0] = ('typeTest1', 'environTest1')
mock_data_struct[1] = ('typeTest1', 'environTest2')
mock_data_struct[2] = ('typeTest2', 'environTest2')
mock_data_struct[3] = ('typeTest1', 'environTest2')
mock_data_struct[4] = ('typeTest1', 'environTest1')

mock_cell_array = {'task': np.array([[
    [],
    [mock_data_struct],
    [mock_data_struct]
]])
}


@patch('src.data_processing.loadmat', return_value=mock_cell_array)
def test_get_epochs(mock_loadmat):
    Animal = namedtuple('Animal', {'directory', 'short_name'})
    animal = Animal(directory='test_dir', short_name='Test')
    day = 2
    expected_length = 5

    assert len(get_epochs(animal, day)) == expected_length


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
