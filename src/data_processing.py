# -*- coding: utf-8 -*-
'''
Functions for accessing data in the Frank lab format
'''

import os
import sys
import itertools
import scipy.io
import numpy as np
import pandas as pd


def get_data_filename(animal, day, file_type):
    '''Returns the Matlab file name assuming it is in the Raw Data directory.
    File type is a string that refers to the various data structure names
    (DIO, tasks, linpos) Animal is a named tuple. Day is an integer giving the
    recording session day.
    '''
    data_dir = os.path.join(os.path.abspath(os.path.pardir), 'Raw-Data')
    filename = '{animal.short_name}{file_type}{day:02d}.mat'.format(
        data_dir=data_dir,
        animal=animal,
        file_type=file_type,
        day=day)
    return os.path.join(data_dir, animal.directory, filename)


def get_epochs(animal, day):
    '''Returns a list of three-element tuples (animal, day, epoch index) that
    can access the data structure within each Matlab file. Epoch type
    is the task for that epoch (sleep, run, etc.) and environment is
    the type of maze the animal is in (if any). If no epoch type or
    environment is given, returns all epoch types and environments.
    Days can be either a single integer day or a list of days.
    '''
    try:
        task_file = scipy.io.loadmat(
            get_data_filename(animal, day, 'task'))
        return [(animal, day, ind + 1) for ind, epoch in enumerate(task_file['task'][0, - 1][0])]
    except IOError as err:
        print('I/O error({0}): {1}'.format(err.errno, err.strerror))
        sys.exit()


def get_data_structure(animal, day, file_type, variable):
    '''Returns a filtered list containing the data structures corresponding to the
    animal, day, file_type, epoch specified.
    '''
    epoch = get_epochs(animal, day)
    file = scipy.io.loadmat(get_data_filename(animal, day, file_type))
    try:
        return [file[variable][0, day - 1][0, ind - 1] for _, day, ind in epoch]
    except IndexError:
        # candripples file doesn't have a cell for the last epoch.
        return [file[variable][0, day - 1][0, ind - 1] for _, day, ind in epoch[:-1]]


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


def get_position_dataframe(epoch_index, animals):
    '''Returns a list of position dataframes with a length corresponding to the number of
    epochs in the epoch index -- either a tuple or a list of tuples with the format
    (animal, day, epoch_number)
    '''
    animal, day, epoch = epoch_index
    epoch_data = get_data_structure(animals[animal], day, 'pos', 'pos')[
        epoch - 1]['data'][0, 0]
    return _convert_position_array_to_dataframe(epoch_data)


def _convert_position_array_to_dataframe(array):
    column_names = ['time', 'x_position', 'y_position', 'head_direction',
                    'speed', 'smoothed_x_position', 'smoothed_y_position',
                    'smoothed_head_direction', 'smoothed_speed']
    drop_columns = ['smoothed_x_position', 'smoothed_y_position',
                    'smoothed_speed', 'smoothed_head_direction']
    return (pd.DataFrame(array, columns=column_names)
            .set_index('time')
            .drop(drop_columns, axis=1))


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
    data_dir = os.path.join(os.path.abspath(os.path.pardir), 'Raw-Data')
    animal, day, epoch_ind, tetrode_number = tetrode_tuple
    filename = '{animal.short_name}eeg{day:02d}-{epoch}-{tetrode_number:02d}.mat'.format(
        data_dir=data_dir,
        animal=animals[animal],
        day=day,
        epoch=epoch_ind,
        tetrode_number=tetrode_number
    )
    return os.path.join(data_dir, animals[animal].directory, 'EEG', filename)


def _get_tetrode_id(dataframe):
    return dataframe.animal + \
        dataframe.day.astype(str) + \
        dataframe.epoch_ind.astype(str) + \
        dataframe.tetrode_number.astype(str)


def convert_tetrode_epoch_to_dataframe(tetrodes_in_epoch, animal, day, epoch_ind):
    '''
    Given an epoch data structure, return a cleaned up DataFrame
    '''
    tetrode_dict_list = [_convert_to_dict(
        tetrode) for tetrode in tetrodes_in_epoch[0][0]]
    return (pd.DataFrame(tetrode_dict_list)
            # convert numcells to integer type
              .assign(numcells=lambda x: x['numcells'].astype(int))
            # convert depth to integer type
              .assign(depth=lambda x: x['depth'].astype(int))
            # convert numcells to integer type
              .assign(area=lambda x: x['area'])
              .assign(animal=lambda x: animal)
              .assign(day=lambda x: day)
              .assign(epoch_ind=lambda x: epoch_ind)
              .assign(tetrode_number=lambda x: x.index + 1)
              .assign(tetrode_id=_get_tetrode_id)
            # set index to identify rows
              .set_index(['animal', 'day', 'epoch_ind', 'tetrode_number'], drop=False)
              .sort_index()
            )


def get_tetrode_info(animal):
    '''Returns the Matlab tetrodeinfo file name assuming it is in the Raw Data directory.
    '''
    data_dir = os.path.join(os.path.abspath(os.path.pardir), 'Raw-Data')
    filename = '{animal.short_name}tetinfo.mat'.format(animal=animal)
    return os.path.join(data_dir, animal.directory, filename)


def _convert_to_dict(struct_array):
    try:
        return {name: np.squeeze(struct_array[name][0, 0][0])
                for name in struct_array.dtype.names}
    except TypeError:
        return {}


def get_LFP_dataframe(tetrode_index, animals):
    ''' Given a tetrode index tuple and the animals dictionary,
    return the LFP data and start time
    '''
    lfp_file = scipy.io.loadmat(get_LFP_filename(tetrode_index, animals))
    lfp_data = lfp_file['eeg'][0, -1][0, -1][0, -1]
    lfp_time = _get_LFP_time(lfp_data['starttime'][0, 0][0],
                             lfp_data['data'][0, 0].size,
                             lfp_data['samprate'][0, 0])
    data_dict = {'time': lfp_time,
                 'electric_potential': lfp_data['data'][0, 0].squeeze()
                 }
    return pd.DataFrame(data_dict).set_index('time').sort_index()


def _get_LFP_time(start_time, number_samples, sampling_frequency):
    ''' Returns an array of time stamps
    '''
    sampling_frequency = int(np.round(sampling_frequency))

    return start_time + np.arange(0, number_samples) * (1 / sampling_frequency)


def get_neuron_info(animal):
    '''Returns the Matlab cellinfo file name assuming it is in the Raw Data directory.
    '''
    data_dir = os.path.join(os.path.abspath(os.path.pardir), 'Raw-Data')
    filename = '{animal.short_name}cellinfo.mat'.format(animal=animal)
    return os.path.join(data_dir, animal.directory, filename)


def _get_neuron_id(dataframe):
    return dataframe.animal + \
        dataframe.day.astype(str) + \
        dataframe.epoch_ind.astype(str) + \
        dataframe.tetrode_number.astype(str) + \
        dataframe.neuron_number.astype(str)


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

    NEURON_INDEX = ['animal', 'day', 'epoch_ind',
                    'tetrode_number', 'neuron_number']

    neuron_dict_list = [_add_to_dict(_convert_to_dict(neuron), tetrode_ind, neuron_ind)
                        for tetrode_ind, tetrode in enumerate(tetrodes_in_epoch[0][0])
                        for neuron_ind, neuron in enumerate(tetrode[0])
                        if neuron.size > 0
                        ]
    return (pd.DataFrame(neuron_dict_list)
              .drop(DROP_COLUMNS, axis=1, errors='ignore')
              .assign(animal=animal)
              .assign(day=day)
              .assign(epoch_ind=epoch_ind)
              .assign(neuron_id=_get_neuron_id)
            # set index to identify rows
              .set_index(NEURON_INDEX, drop=False)
              .sort_index()
            )


def _add_to_dict(dictionary, tetrode_ind, neuron_ind):
    dictionary['tetrode_number'] = tetrode_ind + 1
    dictionary['neuron_number'] = neuron_ind + 1
    return dictionary


def make_epochs_dataframe(animals, days):
    # Get all epochs
    tasks = [(get_data_structure(animals[animal], day, 'task', 'task'), animal, day)
             for animal in animals
             for day in days]
    epochs = [epoch
              for day_structure, _, _ in tasks
              for epoch in day_structure]

    task_dataframe = pd.DataFrame([
        {name: epoch[name][0][0][0]
         for name in epoch.dtype.names
         if name not in 'linearcoord'}
        for epoch in epochs])

    index_labels = pd.DataFrame([{'animal': animal, 'day': day, 'epoch_ind': epoch_ind + 1}
                                 for day_structure, animal, day in tasks
                                 for epoch_ind, epoch in enumerate(day_structure)])

    return (pd.concat([index_labels, task_dataframe], axis=1, join_axes=[task_dataframe.index])
            .set_index(['animal', 'day', 'epoch_ind'], drop=False)
            .sort_index()
            .assign(environment=lambda x: pd.Categorical(x['environment']))
            .assign(type=lambda x: pd.Categorical(x['type'])))


def make_tetrode_dataframe(animals):
    tetrode_file_names = [(get_tetrode_info(animals[animal]), animal)
                          for animal in animals]
    tetrode_data = [(scipy.io.loadmat(file_name[0]), file_name[1])
                    for file_name in tetrode_file_names]

    # Make a dictionary with (animal, day, epoch_ind) as the keys
    return {(animal, day_ind + 1, epoch_ind + 1):
            convert_tetrode_epoch_to_dataframe(
                epoch, animal, day_ind + 1, epoch_ind + 1)
            for info, animal in tetrode_data
            for day_ind, day in enumerate(info['tetinfo'].T)
            for epoch_ind, epoch in enumerate(day[0].T)
            }


def filter_list_by_pandas_series(list_to_filter, pandas_boolean_series):
    ''' Convenience function for filtering a list by the criterion of a pandas series
    Returns a list.
    '''
    is_in_list = list(pandas_boolean_series)
    if len(list_to_filter) != len(is_in_list):
        raise ValueError(
            'list to filter must be the same length as the pandas series')
    return [list_element for list_element, is_in_list in zip(list_to_filter, is_in_list)
            if is_in_list]


def get_spikes_dataframe(neuron_index, animals):
    animal, day, epoch, tetrode_number, neuron_number = neuron_index
    neuron_file = scipy.io.loadmat(
        get_data_filename(animals[animal], day, 'spikes'))
    try:
        neuron_data = neuron_file['spikes'][
            0, -1][0, epoch - 1][0, tetrode_number - 1][0, neuron_number - 1][0]['data'][0][:, 0]
        data_dict = {'time': neuron_data,
                     'is_spike': 1
                     }
    except IndexError:
        data_dict = {'time': [],
                     'is_spike': []}
    return pd.DataFrame(data_dict).set_index('time').sort_index()


def make_neuron_dataframe(animals):
    neuron_file_names = [(get_neuron_info(animals[animal]), animal)
                         for animal in animals]
    neuron_data = [(scipy.io.loadmat(file_name[0]), file_name[1])
                   for file_name in neuron_file_names]
    return {(animal, day_ind + 1, epoch_ind + 1):
            convert_neuron_epoch_to_dataframe(
                epoch, animal, day_ind + 1, epoch_ind + 1)
            for cellfile, animal in neuron_data
            for day_ind, day in enumerate(cellfile['cellinfo'].T)
            for epoch_ind, epoch in enumerate(day[0].T)
            }


def get_interpolated_position_dataframe(epoch_index, animals):
    time = get_trial_time(epoch_index, animals)
    position = (pd.concat([get_linear_position_structure(epoch_index, animals),
                           get_position_dataframe(epoch_index, animals)], axis=1)
                .assign(trajectory_direction=_trajectory_direction)
                .assign(trajectory_turn=_trajectory_turn)
                .assign(trial_number=_trial_number)
                .assign(linear_position=_linear_position)
                )
    categorical_columns = ['trajectory_category_ind',
                           'trajectory_turn', 'trajectory_direction', 'trial_number']
    continuous_columns = ['head_direction', 'speed',
                          'linear_distance', 'linear_position',
                          'x_position', 'y_position']
    position_categorical = (position
                            .drop(continuous_columns, axis=1)
                            .reindex(index=time, method='pad'))
    position_continuous = (position
                           .drop(categorical_columns, axis=1)
                           .dropna())
    new_index = pd.Index(np.sort(np.unique(np.concatenate(
       (position_continuous.index, time)))), name='time')
    interpolated_position = (position_continuous
                             .reindex(index=new_index)
                             .interpolate(method='spline', order=3)
                             .reindex(index=time))
    return (pd.concat([position_categorical, interpolated_position], axis=1)
            .fillna(method='backfill'))


def _linear_position(df):
    is_left_arm = (df.trajectory_category_ind == 1) | (
        df.trajectory_category_ind == 2)
    return np.where(is_left_arm, -1 * df.linear_distance, df.linear_distance)


def _trial_number(df):
    return np.cumsum(df.trajectory_category_ind.diff().fillna(0) > 0) + 1


def _trajectory_turn(df):
    trajectory_turn = {0: np.nan, 1: 'Left', 2: 'Right', 3: 'Left', 4: 'Right'}
    return df.trajectory_category_ind.map(trajectory_turn)


def _trajectory_direction(df):
    trajectory_direction = {0: np.nan, 1: 'Outbound',
                            2: 'Inbound', 3: 'Outbound', 4: 'Inbound'}
    return df.trajectory_category_ind.map(trajectory_direction)


def get_linear_position_structure(epoch_key, animals, trajectory_category=None):
    animal, day, epoch = epoch_key
    struct = get_data_structure(
        animals[animal], day, 'linpos', 'linpos')[epoch - 1][0][0]['statematrix']
    include_fields = ['time', 'traj', 'lindist']
    new_names = {'time': 'time', 'traj': 'trajectory_category_ind',
                 'lindist': 'linear_distance'}
    return (pd.DataFrame(
        {new_names[name]: struct[name][0][0].flatten() for name in struct.dtype.names
         if name in include_fields})
        .set_index('time')
    )


def get_spike_indicator_dataframe(neuron_index, animals):
    ''' Returns a dataframe with a spike time indicator column
    where 1 indicates a spike at that time and 0 indicates no
    spike at that time. The number of datapoints corresponds
    is the same as the LFP.
    '''
    time = get_trial_time(neuron_index, animals)
    spikes_df = get_spikes_dataframe(neuron_index, animals)
    spikes_df.index = time[find_closest_ind(time, spikes_df.index.values)]
    return spikes_df.reindex(index=time, fill_value=0)


def get_trial_time(index, animals):
    try:
        animal, day, epoch, tetrode_number = index[:4]
    except ValueError:
        # no tetrode number provided
        tetrode_info = make_tetrode_dataframe(animals)
        animal, day, epoch, tetrode_number = tetrode_info[index].index[0]
    lfp_df = get_LFP_dataframe((animal, day, epoch, tetrode_number), animals)
    return lfp_df.index


def get_windowed_dataframe(dataframe, segments, window_offset, sampling_frequency):
    segments = iter(segments)
    for segment_start, segment_end in segments:
        # Handle floating point inconsistencies in the index
        segment_start_ind = dataframe.index.get_loc(
            segment_start, method='nearest')
        segment_start = dataframe.iloc[segment_start_ind].name
        if window_offset is not None:
            window_start_ind = np.max([0, int(segment_start_ind +
                                       np.fix(window_offset[0] * sampling_frequency))])
            window_end_ind = np.min([len(dataframe), int(segment_start_ind +
                                     np.fix(window_offset[1] * sampling_frequency)) + 1])
            yield (dataframe
                   .iloc[window_start_ind:window_end_ind, :]
                   .reset_index()
                   .assign(time=lambda x: np.round(x.time - segment_start, decimals=4))
                   .set_index('time'))
        else:
            yield (dataframe.loc[segment_start:segment_end, :]
                            .reset_index()
                            .assign(time=lambda x: np.round(x.time - segment_start, decimals=4))
                            .set_index('time'))


def reshape_to_segments(dataframe, segments, window_offset=None,
                        sampling_frequency=1500, concat_axis=0):
    segment_label = [(segment_ind + 1, segment_start, segment_end)
                     for segment_ind, (segment_start, segment_end)
                     in enumerate(segments)]
    return (pd.concat(list(get_windowed_dataframe(dataframe, segments,
                                                  window_offset, sampling_frequency)),
                      keys=segment_label,
                      names=['segment_number',
                             'segment_start', 'segment_end'],
                      axis=concat_axis)
            .sort_index())


def get_tetrode_pair_info(tetrode_info):
    pair_index = pd.MultiIndex.from_tuples(list(itertools.combinations(tetrode_info.index, 2)),
                                           names=['tetrode1', 'tetrode2'])
    no_rename = {'animal_1': 'animal',
                 'day_1': 'day',
                 'epoch_ind_1': 'epoch_ind'}
    tetrode1 = (tetrode_info
                .loc[pair_index.get_level_values('tetrode1')]
                .reset_index(drop=True)
                .add_suffix('_1')
                .rename(columns=no_rename)
                )
    tetrode2 = (tetrode_info
                .loc[pair_index.get_level_values('tetrode2')]
                .reset_index(drop=True)
                .drop(['animal', 'day', 'epoch_ind'], axis=1)
                .add_suffix('_2')
                )
    return (pd.concat([tetrode1, tetrode2], axis=1)
            .set_index(pair_index))

if __name__ == '__main__':
    sys.exit()
