import pytest
import collections
from unittest.mock import patch
import numpy as np
import src.data_filter as df


def test_data_file_name_returns_correct_file():
    Animal = collections.namedtuple('Animal', {'directory', 'short_name'})
    animal = Animal(directory='test_dir', short_name='Test')
    file_type = 'dummy'

    day = 2
    file_name = df.get_data_filename(animal, day, file_type)
    assert '/Raw-Data/test_dir/Testdummy02.mat' in file_name

    day = 11
    file_name = df.get_data_filename(animal, day, file_type)
    assert '/Raw-Data/test_dir/Testdummy11.mat' in file_name


@pytest.mark.parametrize("search_array, target, expected_index", [
    (np.arange(50, 150), 66, 16),
    (np.arange(50, 150), 45, 0),
    (np.arange(50, 150), 200, 99),
    (np.arange(50, 150), 66.4, 16),
    (np.arange(50, 150), 66.7, 17),
    (np.arange(50, 150), [55, 65, 137], [5, 15, 87]),
])
def test_find_closest_ind(search_array, target, expected_index):
    assert np.all(df.find_closest_ind(search_array, target) == expected_index)

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


@patch('scipy.io.loadmat')
@pytest.mark.parametrize("days, epoch_type, environment, expected_length", [
    (2, '', '', 5),
    (2, 'typeTest1', '', 4),
    (2, 'typeTest2', '', 1),
    (2, 'typeTest1', 'environTest1', 2),
    (2, 'typeTest1', 'environTest2', 2),
    (2, 'typeTest2', 'environTest1', 0),
    (2, 'typeTest2', 'environTest3', 0),
    (2, '', 'environTest2', 3),
    ([2, 3], '', '', 10),
    ([2, 3], 'typeTest1', '', 8),
])
def test_get_epochs(mock_loadmat, days, epoch_type, environment, expected_length):
    Animal = collections.namedtuple('Animal', {'directory', 'short_name'})
    animal = Animal(directory='test_dir', short_name='Test')
    mock_loadmat.return_value = mock_cell_array

    assert len(df.get_epochs(animal, days, epoch_type=epoch_type, environment=environment)) == expected_length
