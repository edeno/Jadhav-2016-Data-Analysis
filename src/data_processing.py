'''Functions for accessing data in the Frank lab format and saving

'''

from glob import glob
from logging import getLogger
from itertools import chain
from os import listdir, makedirs, walk
from os.path import abspath, join, pardir, isfile, dirname
from shutil import copyfile
import re
from sys import exit
from warnings import filterwarnings

import numpy as np
import pandas as pd
from xarray.backends.api import (
    basestring, _CONCAT_DIM_DEFAULT, _default_lock, open_dataset,
    auto_combine, _MultiFileCloser)
from scipy.io import loadmat

logger = getLogger(__name__)

ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')


def get_data_filename(animal, day, file_type):
    '''Returns the Matlab file name assuming it is in the Raw Data
    directory.

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of recording
    file_type : str
        Data structure name (e.g. linpos, dio)

    Returns
    -------
    filename : str
        Path to data file

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
    epochs : list of tuples, shape (n_epochs,)
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
                for ind, epoch in enumerate(task_file['task'][0, -1][0])]
    except IOError as err:
        logger.error('Failed to load file {0}'.format(
            get_data_filename(animal, day, 'task')))
        exit()


def get_data_structure(animal, day, file_type, variable):
    '''Returns data structures corresponding to the animal, day, file_type
    for all epochs

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    day : int
        Day of recording
    file_type : str
        Data structure name (e.g. linpos, dio)
    variable : str
        Variable in data structure

    Returns
    -------
    variable : list, shape (n_epochs,)
        Elements of list are data structures corresponding to variable

    '''
    try:
        file = loadmat(get_data_filename(animal, day, file_type))
    except IOError:
        logger.error('Failed to load file: {0}'.format(
            get_data_filename(animal, day, file_type)))
        exit()
    n_epochs = file[variable][0, -1].size
    return [file[variable][0, -1][0, ind]
            for ind in np.arange(n_epochs)]


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

    Parameters
    ----------
    epoch_key : tuple
        Defines a single epoch (animal, day, epoch)
    animals : dictionary of namedtuples
        Maps animal name to namedtuple with animal file directory

    Returns
    -------
    position : pandas dataframe
        Contains information about the animal's position, head direction,
        and speed.

    '''
    animal, day, epoch = epoch_key
    position_data = get_data_structure(animals[animal], day, 'pos', 'pos')[
        epoch - 1]['data'][0, 0]
    field_names = get_data_structure(animals[animal], day, 'pos', 'pos')[
        epoch - 1]['fields'][0, 0].item().split()
    NEW_NAMES = {'x': 'x_position',
                 'y': 'y_position',
                 'dir': 'head_direction',
                 'vel': 'speed'}
    time_index = pd.Index(
        position_data[:, field_names.index('time')], name='time')
    return (pd.DataFrame(
                position_data, columns=field_names, index=time_index)
            .rename(columns=NEW_NAMES)
            .drop([name for name in field_names
                   if name not in NEW_NAMES], axis=1))


def find_closest_ind(search_array, target):
    '''Finds the index position in the search_array that is closest to the
    target. This works for large search_arrays. See:
    http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python

    Parameters
    ----------
    search_array : ndarray
    target : ndarray element

    Returns
    -------
    index : int
        Index closest to target element.

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


def get_LFP_filename(tetrode_key, animals):
    '''Returns a file name for the tetrode file LFP for an epoch.

    Parameters
    ----------
    tetrode_key : tuple
        Four element tuple with format (animal, day, epoch, tetrode_number)
    animals : dictionary of namedtuples
        Maps animal name to namedtuple with animal file directory

    Returns
    -------
    filename : str
        File path to tetrode file LFP
    '''
    animal, day, epoch, tetrode_number = tetrode_key
    filename = ('{animal.short_name}eeg{day:02d}-{epoch}-'
                '{tetrode_number:02d}.mat').format(
                    data_dir=RAW_DATA_DIR, animal=animals[animal],
                    day=day, epoch=epoch, tetrode_number=tetrode_number
    )
    return join(
        RAW_DATA_DIR, animals[animal].directory, 'EEG', filename)


def _get_tetrode_id(dataframe):
    return (dataframe.animal + '_' +
            dataframe.day.map('{:02d}'.format) + '_' +
            dataframe.epoch.map('{:02}'.format) + '_' +
            dataframe.tetrode_number.map('{:03}'.format))


def convert_tetrode_epoch_to_dataframe(tetrodes_in_epoch, epoch_key):
    '''Convert tetrode information data structure to dataframe.

    Parameters
    ----------
    tetrodes_in_epoch : ?
    epoch_key : tuple

    Returns
    -------
    tetrode_info : dataframe
        Tetrode information
    '''
    animal, day, epoch = epoch_key
    tetrode_dict_list = [_convert_to_dict(
        tetrode) for tetrode in tetrodes_in_epoch[0][0]]
    return (pd.DataFrame(tetrode_dict_list)
              .assign(numcells=lambda x: x['numcells'])
              .assign(depth=lambda x: x['depth'])
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


def get_tetrode_info_path(animal):
    '''Returns the Matlab tetrode info file name assuming it is in the
    Raw Data directory.

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.

    Returns
    -------
    filename : str
        The path to the information about the tetrodes for a given animal.

    '''
    filename = '{animal.short_name}tetinfo.mat'.format(animal=animal)
    return join(RAW_DATA_DIR, animal.directory, filename)


def _convert_to_dict(struct_array):
    try:
        return {name: struct_array[name].item().item()
                for name in struct_array.dtype.names
                if struct_array[name].item().size == 1}
    except TypeError:
        return {}


def get_LFP_dataframe(tetrode_key, animals):
    '''Gets the LFP data for a given epoch and tetrode.

    Parameters
    ----------
    tetrode_key : tuple, (animal, day, epoch, tetrode_number)
    animals : dictionary of namedtuples
        Maps animal name to namedtuple with animal file directory

    Returns
    -------
    LFP : pandas dataframe
        Contains the electric potential and time
    '''
    try:
        lfp_file = loadmat(get_LFP_filename(tetrode_key, animals))
        lfp_data = lfp_file['eeg'][0, -1][0, -1][0, -1]
        lfp_time = _get_LFP_time(lfp_data['starttime'][0, 0].item(),
                                 lfp_data['data'][0, 0].size,
                                 lfp_data['samprate'][0, 0])
        return pd.Series(
            data=lfp_data['data'][0, 0].squeeze(),
            index=lfp_time,
            name='electric_potential')
    except FileNotFoundError:
        pass


def _get_LFP_time(start_time, n_samples, sampling_frequency):
    '''The recording time for a tetrode

    Parameters
    ----------
    start_time : float
        Start time of recording.
    n_samples : int
        Number of samples in recording.
    sampling_frequency : float
        Number of samples per time

    Returns
    -------
    time : pandas Index

    '''
    sampling_frequency = int(np.round(sampling_frequency))

    return pd.Index(
        start_time + np.arange(0, n_samples) / sampling_frequency,
        name='time')


def get_neuron_info_path(animal):
    '''Returns the path to the neuron info matlab file

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.

    Returns
    -------
    path : str

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


def load_task(file_name, animal):
    data = loadmat(file_name, variable_names=('task'))['task']
    day = data.shape[-1]
    epochs = data[0, -1][0]
    n_epochs = len(epochs)
    index = pd.MultiIndex.from_product(
        ([animal.short_name], [day], np.arange(n_epochs) + 1),
        names=['animal', 'day', 'epoch'])

    return pd.DataFrame(
        [{name: epoch[name].item().squeeze()
         for name in epoch.dtype.names
         if name in ['environment', 'exposure', 'type']}
         for epoch in epochs]).set_index(index)


def get_task(animal):
    task_files = glob(join(RAW_DATA_DIR, animal.directory, '*task*.mat'))
    return pd.concat(load_task(task_file, animal)
                     for task_file in task_files)


def make_epochs_dataframe(animals):
    return (
        pd.concat([get_task(animal) for animal in animals.values()])
        .sort_index())


def make_tetrode_dataframe(animals):
    tetrode_file_names = [
        (get_tetrode_info_path(animal), animal.short_name)
        for animal in animals.values()]
    tetrode_data = [(loadmat(file_name), animal)
                    for file_name, animal in tetrode_file_names]

    # Make a dictionary with (animal, day, epoch) as the keys
    return pd.concat(
        [convert_tetrode_epoch_to_dataframe(
                epoch, (animal, day_ind + 1, epoch_ind + 1))
         for info, animal in tetrode_data
         for day_ind, day in enumerate(info['tetinfo'].T)
         for epoch_ind, epoch in enumerate(day[0].T)
         ]).sort_index()


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
    neuron_file_names = [(get_neuron_info_path(animals[animal]), animal)
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
    position_continuous = position.drop(categorical_columns, axis=1)
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
        lfp_df = get_LFP_dataframe(
            (animal, day, epoch, tetrode_number), animals)
    except ValueError:
        # no tetrode number provided
        tetrode_info = (
            make_tetrode_dataframe(animals)
            .loc[key]
            .set_index(['animal', 'day', 'epoch', 'tetrode_number']))
        lfp_df = pd.concat(
            [get_LFP_dataframe(tetrode_key, animals)
             for tetrode_key in tetrode_info.index],
            axis=1)

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
    TO_SECONDS = 1E4
    mark_file = loadmat(get_mark_filename(tetrode_key, animals))
    mark_names = [name[0][0].lower().replace(' ', '_')
                  for name in mark_file['filedata'][0, 0]['paramnames']]
    mark_data = mark_file['filedata'][0, 0]['params']
    mark_data[:, mark_names.index('time')] = mark_data[
        :, mark_names.index('time')] / TO_SECONDS

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
    # NOTE: Using first tetrode time because of a problem with LFP
    # extraction in Bond. In general this is not desirable.
    time = get_trial_time(tetrode_key[:3], animals)
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


def get_analysis_file_path(animal, day, epoch):
    filename = '{animal}_{day:02d}_{epoch:02d}.nc'.format(
        animal=animal, day=day, epoch=epoch)
    return join(PROCESSED_DATA_DIR, filename)


def save_xarray(epoch_key, dataset, group):
    path = get_analysis_file_path(*epoch_key)
    write_mode = 'a' if isfile(path) else 'w'
    dataset.to_netcdf(path=path, group=group, mode=write_mode)


def _open_dataset(*args, **kwargs):
    try:
        return open_dataset(*args, **kwargs)
    except (IndexError, OSError):
        return None


def open_mfdataset(paths, chunks=None, concat_dim=_CONCAT_DIM_DEFAULT,
                   compat='no_conflicts', preprocess=None, engine=None,
                   lock=None, **kwargs):
    '''Open multiple files as a single dataset.

    This function is adapted from the xarray function of the same name.
    The main difference is that instead of failing on files that do not
    exist, this function keeps processing.

    Requires dask to be installed.  Attributes from the first dataset file
    are used for the combined dataset.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an
        explicit list of files to open.
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by
        chunk sizes. In general, these should divide the dimensions of each
        dataset. If int, chunk each dimension by ``chunks``.
        By default, chunks will be chosen to load entire input files into
        memory at once. This has a major impact on performance: please see
        the full documentation for more details.
    concat_dim : None, str, DataArray or Index, optional
        Dimension to concatenate files along. This argument is passed on to
        :py:func:`xarray.auto_combine` along with the dataset objects. You
        only need to provide this argument if the dimension along which you
        want to concatenate is not a dimension in the original datasets,
        e.g., if you want to stack a collection of 2D arrays along a third
        dimension. By default, xarray attempts to infer this argument by
        examining component files. Set ``concat_dim=None`` explicitly to
        disable concatenation.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:
        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
    preprocess : callable, optional
        If provided, call this function on each dataset prior to
        concatenation.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio'}, optional
        Engine to use when reading files. If not provided, the default
        engine is chosen based on available dependencies, with a preference
        for 'netcdf4'.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many
        files being open.  However, this option doesn't work with streams,
        e.g., BytesIO.
    lock : False, True or threading.Lock, optional
        This argument is passed on to :py:func:`dask.array.from_array`. By
        default, a per-variable lock is used when reading data from netCDF
        files with the netcdf4 and h5netcdf engines to avoid issues with
        concurrent access when using dask's multithreaded backend.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.
    Returns
    -------
    xarray.Dataset
    See Also
    --------
    auto_combine
    open_dataset

    '''
    filterwarnings('ignore', 'elementwise comparison failed;')
    filterwarnings('ignore', 'numpy equal will not check object')

    if isinstance(paths, basestring):
        paths = sorted(glob(paths))
    if not paths:
        raise IOError('no files to open')

    if lock is None:
        lock = _default_lock(paths[0], engine)
    datasets = [_open_dataset(p, engine=engine, chunks=chunks or {},
                              lock=lock, **kwargs) for p in paths]
    file_objs = [ds._file_obj for ds in datasets if ds is not None]

    if isinstance(concat_dim, pd.Index):
        name = concat_dim.name
        concat_dim = concat_dim.take(
            [ind for ind, ds in enumerate(datasets) if ds is not None])
        concat_dim.name = name

    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets if ds is not None]

    if concat_dim is _CONCAT_DIM_DEFAULT:
        combined = auto_combine(datasets, compat=compat)
    else:
        combined = auto_combine(datasets, concat_dim=concat_dim,
                                compat=compat)
    combined._file_obj = _MultiFileCloser(file_objs)
    combined.attrs = datasets[0].attrs

    return combined


def read_analysis_files(epoch_keys, **kwargs):
    epoch_keys.name = 'recording_session'
    file_names = [get_analysis_file_path(*epoch_key)
                  for epoch_key in epoch_keys]
    return open_mfdataset(
        file_names, concat_dim=epoch_keys, **kwargs)


def get_group_name(resolution, covariate, level, connectivity_measure):
    return ('/'.join([resolution, covariate, level, connectivity_measure])
            .replace('//', '/'))


def copy_animal(animal, src_directory, target_directory):
    '''

    Parameters
    ----------
    animal : namedtuple
        First element is the directory where the animal's data is located.
        The second element is the animal shortened name.
    src_directory : str
    target_directory : str

    '''
    processed_data_dir = join(src_directory, animal.short_name)
    target_data_dir = join(target_directory, animal.directory)
    try:
        makedirs(target_data_dir)
    except FileExistsError:
        pass

    FILE_TYPES = ['cellinfo', 'linpos', 'pos', 'rawpos', 'task', 'tetinfo',
                  'spikes']
    data_files = [glob(join(processed_data_dir,
                       '{animal.short_name}{file_type}*.mat').format(
                        animal=animal, file_type=file_type))
                  for file_type in FILE_TYPES]
    for old_path in chain.from_iterable(data_files):
        new_path = join(target_data_dir, old_path.split('/')[-1])

        print('Copying {old_path}\nto \n{new_path}\n'.format(
            old_path=old_path,
            new_path=new_path
        ))
        copyfile(old_path, new_path)

    src_lfp_data_dir = join(processed_data_dir, 'EEG')
    target_lfp_data_dir = join(target_data_dir, 'EEG')
    try:
        makedirs(target_lfp_data_dir)
    except FileExistsError:
        pass
    lfp_files = [file for file in listdir(src_lfp_data_dir)
                 if 'gnd' not in file and 'eeg' in file]

    for file_name in lfp_files:
        old_path = join(src_lfp_data_dir, file_name)
        new_path = join(target_lfp_data_dir, file_name)

        print('Copying {old_path}\nto \n{new_path}\n'.format(
            old_path=old_path,
            new_path=new_path
        ))
        copyfile(old_path, new_path)

    marks_directory = join(src_directory, animal.directory)
    mark_files = [join(root, f) for root, _, files in walk(marks_directory)
                  for f in files if f.endswith('_params.mat')
                  and not f.startswith('matclust')]
    new_mark_filenames = [rename_mark_file(mark_file, animal)
                          for mark_file in mark_files]
    for mark_file, new_filename in zip(mark_files, new_mark_filenames):
        mark_path = join(target_lfp_data_dir, new_filename)
        print('Copying {mark_file}\nto \n{new_filename}\n'.format(
            mark_file=mark_file,
            new_filename=mark_path
        ))
        copyfile(mark_file, mark_path)


def rename_mark_file(file_str, animal):
    matched = re.match(
        r'.*(\d.+)-(\d.+)_params.mat', file_str.split('/')[-1])
    try:
        day, tetrode_number = matched.groups()
    except AttributeError:
        matched = re.match(
            r'.*(\d+)-.*-(\d+)_params.mat', file_str.split('/')[-1])
        try:
            day, tetrode_number = matched.groups()
        except AttributeError:
            print(file_str)
            raise

    new_name = ('{animal}marks{day:02d}-{tetrode_number:02d}.mat'.format(
        animal=animal.short_name,
        day=int(day),
        tetrode_number=int(tetrode_number)
    ))

    return new_name
