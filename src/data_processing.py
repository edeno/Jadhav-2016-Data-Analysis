'''Functions for accessing data in the Frank lab format and saving

'''

from glob import glob
from itertools import combinations
from logging import getLogger
from os.path import abspath, join, pardir, isfile, dirname
from sys import exit
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

logger = getLogger(__name__)

ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')


def get_data_filename(animal, day, file_type):
    '''Returns the Matlab file name assuming it is in the Raw Data
    directory. File type is a string that refers to the various data
    structure names (DIO, tasks, linpos) Animal is a named tuple.
    Day is an integer giving the recording session day.
    '''
    filename = '{animal.short_name}{file_type}{day:02d}.mat'.format(
        data_dir=RAW_DATA_DIR,
        animal=animal,
        file_type=file_type,
        day=day)
    return join(RAW_DATA_DIR, animal.directory, filename)


def get_epochs(animal, day):
    '''For a given recording day and animal, get the three-element epoch
    key that uniquely identifys the recording epochs in that day.

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of the recording.

    Returns
    -------
    epochs : list of tuples
         A list of three-element tuples (animal, day, epoch key) that
         uniquely identifys the recording epochs in that day.

    Examples
    --------
    >>> from collections import namedtuple
    >>> Animal = namedtuple('Animal', {'directory', 'short_name'})
    >>> animal = Animal(directory='test_dir', short_name='Test')
    >>> day = 2
    >>> get_epochs(animal, day)

    '''
    try:
        task_file = loadmat(
            get_data_filename(animal, day, 'task'))
        return [(animal, day, ind + 1)
                for ind, epoch in enumerate(task_file['task'][0, - 1][0])]
    except IOError as err:
        logger.error('Failed to load file {0}'.format(
            get_data_filename(animal, day, 'task')))
        exit()


def get_data_structure(animal, day, file_type, variable):
    '''Returns a filtered list containing the data structures corresponding
    to the animal, day, file_type, epoch specified.
    '''
    epoch = get_epochs(animal, day)
    try:
        file = loadmat(get_data_filename(animal, day, file_type))
    except IOError:
        logger.error('Failed to load file: {0}'.format(
            get_data_filename(animal, day, file_type)))
        exit()
    try:
        return [file[variable][0, day - 1][0, ind - 1]
                for _, day, ind in epoch]
    except IndexError:
        # candripples file doesn't have a cell for the last epoch.
        return [file[variable][0, day - 1][0, ind - 1]
                for _, day, ind in epoch[:-1]]


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


def get_position_dataframe(epoch_key, animals):
    '''Returns a list of position dataframes with a length corresponding
     to the number of epochs in the epoch key -- either a tuple or a
    list of tuples with the format (animal, day, epoch_number)
    '''
    animal, day, epoch = epoch_key
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
    adjust = (target - search_array[ind - 1]
              ) < (search_array[ind] - target)
    return ind - adjust


def get_pulse_position_ind(pulse_times, position_times):
    '''Returns the index of a pulse from the DIO data structure in terms of the
    position structure time.
    '''
    # position times from the pos files are already adjusted
    TIME_CONSTANT = 1E4
    return [find_closest_ind(
        position_times, pin_pulse_times[:, 0].flatten() / TIME_CONSTANT)
        for pin_pulse_times in pulse_times]


def get_LFP_filename(tetrode_tuple, animals):
    ''' Given an index tuple (animal, day, epoch, tetrode_number) and the
    animals dictionary return a file name for the tetrode file LFP
    '''
    animal, day, epoch, tetrode_number = tetrode_tuple
    filename = ('{animal.short_name}eeg{day:02d}-{epoch}-'
                '{tetrode_number:02d}.mat').format(
                    data_dir=RAW_DATA_DIR, animal=animals[animal],
                    day=day, epoch=epoch, tetrode_number=tetrode_number
    )
    return join(
        RAW_DATA_DIR, animals[animal].directory, 'EEG', filename)


def _get_tetrode_id(dataframe):
    return dataframe.animal + \
        dataframe.day.astype(str) + \
        dataframe.epoch.astype(str) + \
        dataframe.tetrode_number.astype(str)


def convert_tetrode_epoch_to_dataframe(tetrodes_in_epoch, animal,
                                       day, epoch):
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
              .assign(epoch=lambda x: epoch)
              .assign(tetrode_number=lambda x: x.index + 1)
              .assign(tetrode_id=_get_tetrode_id)
            # set index to identify rows
              .set_index(['animal', 'day', 'epoch', 'tetrode_number'],
                         drop=False)
              .sort_index()
            )


def get_tetrode_info(animal):
    '''Returns the Matlab tetrodeinfo file name assuming it is in the
    Raw Data directory.
    '''
    filename = '{animal.short_name}tetinfo.mat'.format(animal=animal)
    return join(RAW_DATA_DIR, animal.directory, filename)


def _convert_to_dict(struct_array):
    try:
        return {name: np.squeeze(struct_array[name][0, 0][0])
                for name in struct_array.dtype.names}
    except TypeError:
        return {}


def get_LFP_dataframe(tetrode_key, animals):
    ''' Given a tetrode key tuple and the animals dictionary,
    return the LFP data and start time
    '''
    lfp_file = loadmat(get_LFP_filename(tetrode_key, animals))
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

    return start_time + np.arange(
        0, number_samples) * (1 / sampling_frequency)


def get_neuron_info(animal):
    '''Returns the Matlab cellinfo file name assuming it is in the Raw Data
    directory.
    '''
    filename = '{animal.short_name}cellinfo.mat'.format(animal=animal)
    return join(RAW_DATA_DIR, animal.directory, filename)


def _get_neuron_id(dataframe):
    return dataframe.animal + \
        dataframe.day.astype(str) + \
        dataframe.epoch.astype(str) + \
        dataframe.tetrode_number.astype(str) + \
        dataframe.neuron_number.astype(str)


def convert_neuron_epoch_to_dataframe(tetrodes_in_epoch, animal, day,
                                      epoch):
    '''
    Given an neuron data structure, return a cleaned up DataFrame
    '''
    DROP_COLUMNS = ['ripmodtag', 'thetamodtag', 'runripmodtag',
                    'postsleepripmodtag', 'presleepripmodtag',
                    'runthetamodtag', 'ripmodtag2', 'runripmodtag2',
                    'postsleepripmodtag2', 'presleepripmodtag2',
                    'ripmodtype', 'runripmodtype', 'postsleepripmodtype',
                    'presleepripmodtype', 'FStag', 'ripmodtag3',
                    'runripmodtag3', 'ripmodtype3', 'runripmodtype3',
                    'tag', 'typetag', 'runripmodtype2',
                    'tag2', 'ripmodtype2', 'descrip']

    NEURON_INDEX = ['animal', 'day', 'epoch',
                    'tetrode_number', 'neuron_number']

    neuron_dict_list = [_add_to_dict(
        _convert_to_dict(neuron), tetrode_ind, neuron_ind)
        for tetrode_ind, tetrode in enumerate(
        tetrodes_in_epoch[0][0])
        for neuron_ind, neuron in enumerate(tetrode[0])
        if neuron.size > 0
    ]
    return (pd.DataFrame(neuron_dict_list)
              .drop(DROP_COLUMNS, axis=1, errors='ignore')
              .assign(animal=animal)
              .assign(day=day)
              .assign(epoch=epoch)
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
    tasks = [(get_data_structure(
        animals[animal], day, 'task', 'task'), animal, day)
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

    index_labels = pd.DataFrame([{'animal': animal,
                                  'day': day,
                                  'epoch': epoch_ind + 1}
                                 for day_structure, animal, day in tasks
                                 for epoch_ind, _ in enumerate(
        day_structure)])

    return (pd.concat(
        [index_labels, task_dataframe],
        axis=1, join_axes=[task_dataframe.index])
        .set_index(['animal', 'day', 'epoch'], drop=False)
        .sort_index()
        .assign(environment=lambda x: pd.Categorical(x['environment']))
        .assign(type=lambda x: pd.Categorical(x['type'])))


def make_tetrode_dataframe(animals):
    tetrode_file_names = [(get_tetrode_info(animals[animal]), animal)
                          for animal in animals]
    tetrode_data = [(loadmat(file_name[0]), file_name[1])
                    for file_name in tetrode_file_names]

    # Make a dictionary with (animal, day, epoch) as the keys
    return {(animal, day_ind + 1, epoch_ind + 1):
            convert_tetrode_epoch_to_dataframe(
                epoch, animal, day_ind + 1, epoch_ind + 1)
            for info, animal in tetrode_data
            for day_ind, day in enumerate(info['tetinfo'].T)
            for epoch_ind, epoch in enumerate(day[0].T)
            }


def filter_list_by_pandas_series(list_to_filter, pandas_boolean_series):
    ''' Convenience function for filtering a list by the criterion
    of a pandas series. Returns a list.
    '''
    is_in_list = list(pandas_boolean_series)
    if len(list_to_filter) != len(is_in_list):
        raise ValueError(
            'list to filter must be the same length as the pandas series')
    return [list_element
            for list_element, is_in_list in zip(list_to_filter, is_in_list)
            if is_in_list]


def get_spikes_dataframe(neuron_key, animals):
    animal, day, epoch, tetrode_number, neuron_number = neuron_key
    neuron_file = loadmat(
        get_data_filename(animals[animal], day, 'spikes'))
    try:
        neuron_data = neuron_file['spikes'][0, -1][0, epoch - 1][
            0, tetrode_number - 1][0, neuron_number - 1][0]['data'][0][
            :, 0]
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
    neuron_data = [(loadmat(file_name[0]), file_name[1])
                   for file_name in neuron_file_names]
    return {(animal, day_ind + 1, epoch_ind + 1):
            convert_neuron_epoch_to_dataframe(
                epoch, animal, day_ind + 1, epoch_ind + 1)
            for cellfile, animal in neuron_data
            for day_ind, day in enumerate(cellfile['cellinfo'].T)
            for epoch_ind, epoch in enumerate(day[0].T)
            }


def get_interpolated_position_dataframe(epoch_key, animals):
    time = get_trial_time(epoch_key, animals)
    position = (pd.concat(
        [get_linear_position_structure(epoch_key, animals),
         get_position_dataframe(epoch_key, animals)], axis=1)
        .assign(trajectory_direction=_trajectory_direction)
        .assign(trajectory_turn=_trajectory_turn)
        .assign(trial_number=_trial_number)
        .assign(linear_position=_linear_position)
    )
    categorical_columns = ['trajectory_category_ind',
                           'trajectory_turn', 'trajectory_direction',
                           'trial_number']
    continuous_columns = ['head_direction', 'speed',
                          'linear_distance', 'linear_position',
                          'x_position', 'y_position']
    position_categorical = (position
                            .drop(continuous_columns, axis=1)
                            .reindex(index=time, method='pad'))
    position_continuous = (position
                           .drop(categorical_columns, axis=1)
                           .dropna())
    new_index = pd.Index(np.unique(np.concatenate(
        (position_continuous.index, time))), name='time')
    interpolated_position = (position_continuous
                             .reindex(index=new_index)
                             .interpolate(method='spline', order=3)
                             .reindex(index=time))
    interpolated_position.loc[
        interpolated_position.linear_distance < 0, 'linear_distance'] = 0
    interpolated_position.loc[interpolated_position.speed < 0, 'speed'] = 0
    return (pd.concat([position_categorical, interpolated_position],
                      axis=1)
            .fillna(method='backfill'))


def _linear_position(df):
    is_left_arm = (df.trajectory_category_ind == 1) | (
        df.trajectory_category_ind == 2)
    return np.where(
        is_left_arm, -1 * df.linear_distance, df.linear_distance)


def _trial_number(df):
    return np.cumsum(df.trajectory_category_ind.diff().fillna(0) > 0) + 1


def _trajectory_turn(df):
    trajectory_turn = {0: np.nan, 1: 'Left',
                       2: 'Right', 3: 'Left', 4: 'Right'}
    return df.trajectory_category_ind.map(trajectory_turn)


def _trajectory_direction(df):
    trajectory_direction = {0: np.nan, 1: 'Outbound',
                            2: 'Inbound', 3: 'Outbound', 4: 'Inbound'}
    return df.trajectory_category_ind.map(trajectory_direction)


def get_linear_position_structure(epoch_key, animals,
                                  trajectory_category=None):
    animal, day, epoch = epoch_key
    struct = get_data_structure(
        animals[animal], day, 'linpos', 'linpos')[epoch - 1][0][0][
            'statematrix']
    include_fields = ['time', 'traj', 'lindist']
    new_names = {'time': 'time', 'traj': 'trajectory_category_ind',
                 'lindist': 'linear_distance'}
    return (pd.DataFrame(
        {new_names[name]: struct[name][0][0].flatten()
         for name in struct.dtype.names
         if name in include_fields})
        .set_index('time')
    )


def get_spike_indicator_dataframe(neuron_key, animals):
    ''' Returns a dataframe with a spike time indicator column
    where 1 indicates a spike at that time and 0 indicates no
    spike at that time. The number of datapoints corresponds
    is the same as the LFP.
    '''
    time = get_trial_time(neuron_key, animals)
    spikes_df = get_spikes_dataframe(neuron_key, animals)
    spikes_df.index = time[find_closest_ind(time, spikes_df.index.values)]
    return spikes_df.reindex(index=time, fill_value=0)


def get_trial_time(key, animals):
    try:
        animal, day, epoch, tetrode_number = key[:4]
    except ValueError:
        # no tetrode number provided
        tetrode_info = make_tetrode_dataframe(animals)
        animal, day, epoch, tetrode_number = tetrode_info[key].index[0]
    lfp_df = get_LFP_dataframe(
        (animal, day, epoch, tetrode_number), animals)
    return lfp_df.index


def get_windowed_dataframe(dataframe, segments, window_offset,
                           sampling_frequency):
    segments = iter(segments)
    for segment_start, segment_end in segments:
        # Handle floating point inconsistencies in the index
        segment_start_ind = dataframe.index.get_loc(
            segment_start, method='nearest')
        segment_start = dataframe.iloc[segment_start_ind].name
        if window_offset is not None:
            window_start_ind = np.max(
                [0, int(segment_start_ind + np.fix(
                    window_offset[0] * sampling_frequency))])
            window_end_ind = np.min(
                [len(dataframe),
                 int(segment_start_ind +
                     np.fix(window_offset[1] * sampling_frequency)) + 1])
            yield (dataframe
                   .iloc[window_start_ind:window_end_ind, :]
                   .reset_index()
                   .assign(time=lambda x: np.round(
                       x.time - segment_start, decimals=4))
                   .set_index('time'))
        else:
            yield (dataframe.loc[segment_start:segment_end, :]
                            .reset_index()
                            .assign(time=lambda x: np.round(
                                x.time - segment_start, decimals=4))
                            .set_index('time'))


def reshape_to_segments(dataframe, segments, window_offset=None,
                        sampling_frequency=1500, concat_axis=0):
    segment_label = [(segment_ind + 1, segment_start, segment_end)
                     for segment_ind, (segment_start, segment_end)
                     in enumerate(segments)]
    return (pd.concat(
        list(get_windowed_dataframe(
            dataframe, segments, window_offset, sampling_frequency)),
        keys=segment_label,
        names=['segment_number', 'segment_start', 'segment_end'],
        axis=concat_axis).sort_index())


def make_tetrode_pair_info(tetrode_info):
    pair_keys = pd.MultiIndex.from_tuples(
        list(combinations(tetrode_info.index, 2)),
        names=['tetrode1', 'tetrode2'])
    no_rename = {'animal_1': 'animal',
                 'day_1': 'day',
                 'epoch_1': 'epoch'}
    tetrode1 = (tetrode_info
                .loc[pair_keys.get_level_values('tetrode1')]
                .reset_index(drop=True)
                .add_suffix('_1')
                .rename(columns=no_rename)
                )
    tetrode2 = (tetrode_info
                .loc[pair_keys.get_level_values('tetrode2')]
                .reset_index(drop=True)
                .drop(['animal', 'day', 'epoch'], axis=1)
                .add_suffix('_2')
                )
    return (pd.concat([tetrode1, tetrode2], axis=1)
            .set_index(pair_keys))


def make_area_pair_info(tetrode_info, epoch_key):
    area_pairs = area_pairs = list(combinations(
        sorted(tetrode_info.area.unique()), 2))
    return (pd.DataFrame(area_pairs, columns=['area_1', 'area_2'])
            .assign(animal=epoch_key[0],
                    day=epoch_key[1],
                    epoch=epoch_key[2])
            .set_index(['animal', 'day', 'epoch', 'area_1', 'area_2'],
                       drop=False))


def get_mark_dataframe(tetrode_key, animals):
    '''Retrieve the marks for each tetrode given a tetrode key

    Parameters
    ----------
    tetrode_key : tuple
        Elements are (animal_short_name, day, epoch, tetrode_number)
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    mark_dataframe : pandas dataframe
        The dataframe index is the time at which the mark occurred
        (in seconds). THe other values are values that can be used as the
        marks.
    '''
    TIME_CONSTANT = 1E4
    mark_file = loadmat(get_mark_filename(tetrode_key, animals))
    mark_names = [name[0][0].lower().replace(' ', '_')
                  for name in mark_file['filedata'][0, 0]['paramnames']]
    mark_data = mark_file['filedata'][0, 0]['params']
    mark_data[:, mark_names.index('time')] = mark_data[
        :, mark_names.index('time')] / TIME_CONSTANT

    return pd.DataFrame(mark_data, columns=mark_names).set_index('time')


def get_mark_filename(tetrode_key, animals):
    '''Given a tetrode key (animal, day, epoch, tetrode_number) and the
    animals dictionary return a file name for the tetrode file marks
    '''
    animal, day, _, tetrode_number = tetrode_key
    filename = ('{animal.short_name}marks{day:02d}-'
                '{tetrode_number:02d}.mat').format(
        data_dir=RAW_DATA_DIR,
        animal=animals[animal],
        day=day,
        tetrode_number=tetrode_number
    )
    return join(
        RAW_DATA_DIR, animals[animal].directory, 'EEG', filename)


def get_mark_indicator_dataframe(tetrode_key, animals):
    time = get_trial_time(tetrode_key, animals)
    mark_dataframe = (get_mark_dataframe(tetrode_key, animals)
                      .loc[time.min():time.max()])
    mark_dataframe.index = time[
        find_closest_ind(time, mark_dataframe.index.values)]
    return mark_dataframe.reindex(index=time, fill_value=np.nan)


def _get_computed_ripple_times(tetrode_tuple, animals):
    '''Returns a list of tuples for a given tetrode in the format
    (start_index, end_index). The indexes are relative
    to the trial time for that session. Data is extracted from the ripples
    data structure and calculated according to the Frank Lab criterion.
    '''
    animal, day, epoch, tetrode_number = tetrode_tuple
    ripples_data = get_data_structure(
        animals[animal], day, 'ripples', 'ripples')
    return zip(
        ripples_data[epoch - 1][0][tetrode_number - 1]['starttime'][
            0, 0].flatten(),
        ripples_data[epoch - 1][0][tetrode_number - 1]['endtime'][
            0, 0].flatten())


def _convert_ripple_times_to_dataframe(ripple_times, dataframe):
    '''Given a list of ripple time tuples (ripple #, start time, end time)
    and a dataframe with a time index (such as the lfp dataframe), returns
    a pandas dataframe with a column with the timestamps of each ripple
    labeled according to the ripple number. Non-ripple times are marked as
    NaN.
    '''
    try:
        index_dataframe = dataframe.drop(dataframe.columns, axis=1)
    except AttributeError:
        index_dataframe = dataframe[0].drop(dataframe[0].columns, axis=1)
    ripple_dataframe = (pd.concat(
        [index_dataframe.loc[start_time:end_time].assign(
            ripple_number=number)
         for number, start_time, end_time in ripple_times]))
    try:
        ripple_dataframe = pd.concat(
            [dataframe, ripple_dataframe], axis=1,
            join_axes=[index_dataframe.index])
    except TypeError:
        ripple_dataframe = pd.concat(
            [pd.concat(dataframe, axis=1), ripple_dataframe],
            axis=1, join_axes=[index_dataframe.index])
    return ripple_dataframe


def get_computed_ripples_dataframe(tetrode_key, animals):
    '''Given a tetrode key (animal, day, epoch, tetrode #), returns a
    pandas dataframe with the pre-computed ripples from the Frank lab
     labeled according to the ripple number. Non-ripple times are marked as
     NaN.
    '''
    ripple_times = _get_computed_ripple_times(tetrode_key, animals)
    [(ripple_ind + 1, start_time, end_time)
     for ripple_ind, (start_time, end_time) in enumerate(ripple_times)]
    lfp_dataframe = get_LFP_dataframe(tetrode_key, animals)
    return (_convert_ripple_times_to_dataframe(ripple_times, lfp_dataframe)
            .assign(
                ripple_indicator=lambda x: x.ripple_number.fillna(0) > 0))


def get_computed_consensus_ripple_times(epoch_key, animals):
    '''Returns a list of tuples for a given epoch in the format
    (start_time, end_time).
    '''
    animal, day, epoch = epoch_key
    ripples_data = get_data_structure(
        animals[animal], day, 'candripples', 'candripples')
    return list(map(tuple, ripples_data[epoch - 1]['riptimes'][0][0]))


def get_lfps_by_area(area, tetrode_info, lfps):
    '''Returns a Pandas Panel of lfps with shape: (lfps x time x trials)'''
    return pd.Panel({tetrode_key: lfps[tetrode_key]
                     for tetrode_key
                     in tetrode_info[tetrode_info.area == area].index})


def get_analysis_file_path(animal, day, epoch):
    filename = '{animal}_{day:02d}_{epoch:02d}.nc'.format(
        animal=animal, day=day, epoch=epoch)
    return join(PROCESSED_DATA_DIR, filename)


def save_multitaper_parameters(epoch_key, multitaper_parameter_name,
                               multitaper_parameters):
    coherence_node = '/{multitaper_parameter_name}'.format(
        multitaper_parameter_name=multitaper_parameter_name)
    with pd.HDFStore(get_analysis_file_path(*epoch_key)) as store:
        (store.get_node(coherence_node)
         ._v_attrs.multitaper_parameters) = multitaper_parameters


def save_ripple_info(epoch_key, ripple_info):
    save_xarray(epoch_key, ripple_info.to_xarray(), '/ripple_info')


def save_tetrode_info(epoch_key, tetrode_info):
    with pd.HDFStore(get_analysis_file_path(*epoch_key)) as store:
        with catch_warnings():
            simplefilter('ignore')
            store.put('/tetrode_info', tetrode_info)


def save_xarray(epoch_key, dataset, group):
    path = get_analysis_file_path(*epoch_key)
    write_mode = 'a' if isfile(path) else 'w'
    dataset.to_netcdf(path=path, group=group, mode=write_mode)


def get_all_tetrode_info():
    '''Retrieves all the hdf5 files from the Processed Data directory
    and returns the tetrode pair info dataframe'''
    file_path = join(PROCESSED_DATA_DIR, '*.nc')
    hdf5_files = glob(file_path)
    return pd.concat([pd.read_hdf(filename, key='/tetrode_info')
                      for filename in hdf5_files]).sort_index()


def get_all_ripple_info():
    '''Retrieves all the hdf5 files from the Processed Data directory
    and returns the tetrode pair info dataframe'''
    file_path = join(PROCESSED_DATA_DIR, '*.nc')
    hdf5_files = glob(file_path)
    return pd.concat([pd.read_hdf(filename, key='/ripple_info')
                      for filename in hdf5_files]).sort_index()


def get_ripple_info(epoch_key):
    '''Retrieves ripple info dataframe given an epoch'''
    file_name = '{}_{:02d}_{:02d}.nc'.format(*epoch_key)
    file_path = join(PROCESSED_DATA_DIR, file_name)
    return pd.read_hdf(file_path, key='/ripple_info')


def read_netcdfs(files, dim, transform_func=None, group=None):
    '''Load multiple netcdf files, concatenate along a dimension, and
    transform the dataset if desired.

    Parameters
    ----------
    files : str or list of str
    dim : str
    transform_func : function
    group : str

    Returns
    -------
    combined_dataset : xarray dataset or dataarray

    References
    ----------
     .. [1] http://xarray.pydata.org/en/stable/io.html

    '''
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path, group=group) as ds:
            logger.debug('Path: {path} \n\t group: {group}'.format(
                path=path, group=group))
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            try:
                ds.load()
                return ds
            except IndexError:
                logger.warn('Selection not found. \n'
                            'Path: {path} \n\t group: {group}'.format(
                                path=path, group=group))

    paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in paths]
    return xr.concat(datasets, dim)
