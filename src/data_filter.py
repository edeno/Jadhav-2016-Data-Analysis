'''
Functions for accessing data in the Frank lab format
'''

import os
import sys
import scipy.io
import numpy as np


def get_data_filename(animal, day, file_type):
    '''Returns the Matlab file name assuming it is in the Raw Data directory.
    File type is a string that refers to the various data structure names
    (DIO, tasks, linpos) Animal is a named tuple. Day is an integer giving the
    recording session day.
    '''
    data_dir = '{working_dir}/Raw-Data'.format(working_dir=os.path.abspath(os.path.pardir))
    return '{data_dir}/{animal.directory}/{animal.short_name}{file_type}{day:02d}.mat'.format(
        data_dir=data_dir,
        animal=animal,
        file_type=file_type,
        day=day)


def get_epochs(animal, days, epoch_type='', environment=''):
    '''Returns a list of three-element tuples (animal, day, epoch index) that
    can access the data structure within each Matlab file. Epoch type
    is the task for that epoch (sleep, run, etc.) and environment is
    the type of maze the animal is in (if any). If no epoch type or
    environment is given, returns all epoch types and environments.
    Days can be either a single integer day or a list of days.
    '''
    epoch_index = list()

    if isinstance(days, int):
        days = [days]

    for day in days:
        try:
            task_file = scipy.io.loadmat(get_data_filename(animal, day, 'task'))
            filtered_epochs = [(ind, epoch) for ind, epoch in enumerate(task_file['task'][0, day - 1][0])
                               if epoch['type'] == epoch_type or
                               epoch_type == '']
            epoch_index += [(animal, day, ind) for ind, epoch in filtered_epochs
                            if ('environment' in epoch.dtype.names and
                            epoch['environment'] == environment) or
                            environment == '']
        except IOError as err:
            print('I/O error({0}): {1}'.format(err.errno, err.strerror))
            sys.exit()

    return epoch_index


def get_data_structure(animal, days, file_type, variable, epoch_type='', environment=''):
    '''Returns a filtered list containing the data structures corresponding to the
    animal, day, file_type, epoch specified.
    '''
    if isinstance(days, int):
        days = [days]
    epoch = get_epochs(animal, days, epoch_type=epoch_type, environment=environment)
    files = {day: scipy.io.loadmat(get_data_filename(animal, day, file_type))
             for day in days}
    return [files[day][variable][0, day - 1][0, ind] for _, day, ind in epoch]


def get_DIO_variable(animal, days, dio_var, epoch_type='', environment=''):
    '''Returns a list of lists given a DIO variable (pulsetimes, timesincelast,
    pulselength, and pulseind) with a length corresponding to the number of
    epochs (first level) and the number of active pins (second level)
    '''
    epoch_pins = get_data_structure(animal, days, 'DIO', 'DIO', epoch_type=epoch_type, environment=environment)
    return [
            [pin[0][dio_var][0][0] for pin in pins.T
             if pin[0].dtype.names is not None]
            for pins in epoch_pins
            ]


def get_position_variables(animal, days, pos_var, epoch_type='', environment=''):
    '''Returns a list of position variable (time, x, y, dir, vel, x-sm,
    y-sm, dir-sm, and vel-sm) arrays with a length corresponding to the number of
    epochs (first level)
    '''
    field_names = ['time', 'x', 'y', 'dir', 'vel', 'x-sm', 'y-sm', 'dir-sm', 'vel-sm']
    field_ind = [field_names.index(var) for var in pos_var]
    epoch_pos = get_data_structure(animal, days, 'pos', 'pos', epoch_type=epoch_type, environment=environment)
    return [pos['data'][0, 0][:, field_ind]
            for pos in epoch_pos]


def find_closest_ind(search_array, target):
    '''Finds the index position in the search_array that is closest to the
    target. This works for large search_arrays. See:
    http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    '''
    ind = search_array.searchsorted(target)  # Get insertion index, need to be sorted
    ind = np.clip(ind, 1, len(search_array) - 1)  # If index is out of bounds, move to bounds
    # Adjust if the left or right index is closer
    adjust = (target - search_array[ind - 1]) < (search_array[ind] - target)
    return ind - adjust


def get_pulse_position_ind(pulse_times, position_times):
    '''Returns the index of a pulse from the DIO data structure in terms of the
    position structure time.
    '''
    TIME_CONSTANT = 1E4  # position times from the pos files are already adjusted
    return [find_closest_ind(position_times, pin_pulse_times[:, 0].flatten() / TIME_CONSTANT)
            for pin_pulse_times in pulse_times]


if __name__ == '__main__':
    sys.exit()
