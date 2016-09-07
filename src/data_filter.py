# -*- coding: utf-8 -*-
'''
Functions for accessing data in the Frank lab format
'''

import os
import sys
import scipy.io
import numpy as np
import pandas as pd


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
            filtered_epochs = [(ind, epoch) for ind, epoch in enumerate(task_file['task'][0, - 1][0])
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
    epoch_pins = get_data_structure(animal, days, 'DIO', 'DIO',
                                    epoch_type=epoch_type,
                                    environment=environment)
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
    epoch_pos = get_data_structure(animal, days, 'pos', 'pos',
                                   epoch_type=epoch_type,
                                   environment=environment)
    return [pos['data'][0, 0][:, field_ind]
            for pos in epoch_pos]


def find_closest_ind(search_array, target):
    '''Finds the index position in the search_array that is closest to the
    target. This works for large search_arrays. See:
    http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    '''
    # Get insertion index, need to be sorted
    ind = search_array.searchsorted(target)
    # If index is out of bounds, move to bounds
    ind = np.clip(ind, 1, len(search_array) - 1)
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


def get_LFP_filename(tetrode_tuple, animals):
    ''' Given an index tuple (animal, day, epoch, tetrode_number) and the animals dictionary
    return a file name for the tetrode file LFP
    '''
    data_dir = '{working_dir}/Raw-Data'.format(working_dir=os.path.abspath(os.path.pardir))
    animal, day, epoch_ind, tetrode_number = tetrode_tuple
    return '{data_dir}/{animal.directory}/EEG/{animal.short_name}eeg{day:02d}-{epoch}-{tetrode_number:02d}.mat'.format(
        data_dir=data_dir,
        animal=animals[animal],
        day=day,
        epoch=epoch_ind,
        tetrode_number=tetrode_number
    )


def convert_tetrode_epoch_to_dataframe(tetrodes_in_epoch, animal, day, epoch_ind):
    '''
    Given an epoch data structure, return a cleaned up DataFrame
    '''
    tetrode_dict_list = [_convert_to_dict(tetrode) for tetrode in tetrodes_in_epoch[0][0]]
    return (pd.DataFrame(tetrode_dict_list)
              .assign(numcells=lambda x: x['numcells'].astype(int))  # convert numcells to integer type
              .assign(depth=lambda x: x['depth'].astype(int))  # convert depth to integer type
              .assign(area=lambda x: pd.Categorical(x['area']))  # convert numcells to integer type
              .assign(animal=lambda x: animal)
              .assign(day=lambda x: day)
              .assign(epoch_ind=lambda x: epoch_ind)
              .assign(tetrode_number=lambda x: x.index + 1)
              .set_index(['animal', 'day', 'epoch_ind', 'tetrode_number'])  # set index to identify rows
              .sort_index()
            )


def get_tetrode_info(animal):
    '''Returns the Matlab tetrodeinfo file name assuming it is in the Raw Data directory.
    '''
    data_dir = '{working_dir}/Raw-Data'.format(working_dir=os.path.abspath(os.path.pardir))
    return '{data_dir}/{animal.directory}/{animal.short_name}tetinfo.mat'.format(
        data_dir=data_dir,
        animal=animal)


def _convert_to_dict(struct_array):
    try:
        return {name: np.squeeze(struct_array[name][0, 0][0])
                for name in struct_array.dtype.names}
    except TypeError:
        return {}


def get_dataframe_index(data_frame):
    ''' Converts pandas dataframe to a list of tuples corresponding to
    the dataframe multi-index
    '''
    index = list(data_frame.index.get_values())
    return index


def _get_LFP_dataframe(tetrode_index, animals):
    ''' Given a tetrode index tuple and the animals dictionary,
    return the LFP data and start time
    '''
    lfp_file = scipy.io.loadmat(get_LFP_filename(tetrode_index, animals))
    lfp_data = lfp_file['eeg'][0, -1][0, -1][0, -1]
    lfp_time = _get_LFP_time(lfp_data['starttime'][0, 0],
                             lfp_data['data'][0, 0].size,
                             lfp_data['samprate'][0, 0])
    data_dict = {'time': lfp_time,
                 'electric_potential': lfp_data['data'][0, 0].squeeze()
                 }
    return pd.DataFrame(data_dict).set_index('time').sort_index()


def get_LFP_data(tetrode_index, animals):
    ''' Given a list of tetrode index tuples and the animals dictionary,
    return a list of the LFP data and start time
    '''
    return [_get_LFP_dataframe(tetrode, animals) for tetrode in tetrode_index]


def _get_LFP_time(start_time, number_samples, sampling_rate):
    ''' Returns an array of time stamps
    '''
    end_time = start_time + (number_samples / sampling_rate)
    return np.round(np.arange(start_time, end_time, (1 / sampling_rate)),
                    decimals=4)


def get_neuron_info(animal):
    '''Returns the Matlab tetrodeinfo file name assuming it is in the Raw Data directory.
    '''
    data_dir = '{working_dir}/Raw-Data'.format(working_dir=os.path.abspath(os.path.pardir))
    return '{data_dir}/{animal.directory}/{animal.short_name}cellinfo.mat'.format(
        data_dir=data_dir,
        animal=animal)


def convert_neuron_epoch_to_dataframe(tetrodes_in_epoch, animal, day, epoch_ind):
    '''
    Given an neuron data structure, return a cleaned up DataFrame
    '''
    DROP_COLUMNS = ['ripmodtag', 'thetamodtag', 'runripmodtag', 'postsleepripmodtag',
                    'presleepripmodtag', 'runthetamodtag', 'ripmodtag2', 'runripmodtag2',
                    'postsleepripmodtag2', 'presleepripmodtag2', 'ripmodtype',
                    'runripmodtype', 'postsleepripmodtype', 'presleepripmodtype',
                    'FStag', 'ripmodtag3', 'runripmodtag3', 'ripmodtype3', 'runripmodtype3',
                    'tag', 'typetag', 'runripmodtype2', 'tag2', 'ripmodtype2', 'descrip']

    NEURON_INDEX = ['animal', 'day', 'epoch_ind', 'tetrode_number', 'neuron_number']

    neuron_dict_list = [_add_to_dict(_convert_to_dict(neuron), tetrode_ind, neuron_ind)
                        for tetrode_ind, tetrode in enumerate(tetrodes_in_epoch[0][0])
                        for neuron_ind, neuron in enumerate(tetrode[0])
                        if neuron.size > 0
                        ]
    return (pd.DataFrame(neuron_dict_list)
              .drop(DROP_COLUMNS, 1, errors='ignore')
              .assign(animal=lambda x: animal)
              .assign(day=lambda x: day)
              .assign(epoch_ind=lambda x: epoch_ind)
              .set_index(NEURON_INDEX)  # set index to identify rows
              .sort_index()
            )


def _add_to_dict(dictionary, tetrode_ind, neuron_ind):
    dictionary['tetrode_number'] = tetrode_ind + 1
    dictionary['neuron_number'] = neuron_ind + 1
    return dictionary


def make_epochs_dataframe(animals, days):
    # Get all epochs
    tasks = [(get_data_structure(animals[animal], day, 'task', 'task'), animal)
             for animal in animals
             for day in days]
    epochs = [(epoch, animal) for day, animal in tasks for epoch in day]

    # Convert into pandas dataframes
    task_data = [
                 {name: epoch[0][name][0][0][0]
                  for name in epoch[0].dtype.names
                  if name not in 'linearcoord'}
                 for epoch in epochs]
    task_dataframe = pd.DataFrame(task_data)

    day_epoch_ind = [{'animal': day[1], 'day': day_ind + 1, 'epoch_ind': epoch_ind + 1}
                     for day_ind, day in enumerate(tasks)
                     for epoch_ind, epoch in enumerate(day[0])]

    day_epoch_dataframe = pd.DataFrame(day_epoch_ind)

    return (pd
            .concat([day_epoch_dataframe, task_dataframe],
                    axis=1,
                    join_axes=[task_dataframe.index])
            .set_index(['animal', 'day', 'epoch_ind'])
            .sort_index()
            .assign(environment=lambda x: pd.Categorical(x['environment']))
            .assign(type=lambda x: pd.Categorical(x['type']))
            )


def make_tetrode_dataframe(animals):
    tetrode_file_names = [(get_tetrode_info(animals[animal]), animal)
                          for animal in animals]
    tetrode_data = [(scipy.io.loadmat(file_name[0]), file_name[1])
                    for file_name in tetrode_file_names]

    # Make a dictionary with (animal, day, epoch_ind) as the keys
    return {(animal, day_ind + 1, epoch_ind + 1):
            convert_tetrode_epoch_to_dataframe(epoch, animal, day_ind + 1, epoch_ind + 1)
            for info, animal in tetrode_data
            for day_ind, day in enumerate(info['tetinfo'].T)
            for epoch_ind, epoch in enumerate(day[0].T)
            }


if __name__ == '__main__':
    sys.exit()
