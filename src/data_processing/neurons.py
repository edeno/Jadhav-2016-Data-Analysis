import numpy as np
import pandas as pd
from scipy.io import loadmat
from os.path import join
from .core import (get_data_filename, _convert_to_dict,
                   RAW_DATA_DIR, logger)
from .tetrodes import get_trial_time


def make_neuron_dataframe(animals):
    neuron_file_names = [(get_neuron_info_path(animals[animal]), animal)
                         for animal in animals]
    neuron_data = [(loadmat(file_name[0]), file_name[1])
                   for file_name in neuron_file_names]
    return pd.concat([
            convert_neuron_epoch_to_dataframe(
                epoch, animal, day_ind + 1, epoch_ind + 1)
            for cellfile, animal in neuron_data
            for day_ind, day in enumerate(cellfile['cellinfo'].T)
            for epoch_ind, epoch in enumerate(day[0].T)
            ]).sort_index()


def get_spikes_dataframe(neuron_key, animals):
    animal, day, epoch, tetrode_number, neuron_number = neuron_key
    neuron_file = loadmat(
        get_data_filename(animals[animal], day, 'spikes'))
    try:
        spike_time = neuron_file['spikes'][0, -1][0, epoch - 1][
            0, tetrode_number - 1][0, neuron_number - 1][0]['data'][0][
            :, 0]
        data_dict = {'time': spike_time,
                     'is_spike': 1
                     }
    except IndexError:
        data_dict = {'time': [],
                     'is_spike': []}
    return pd.DataFrame(data_dict).set_index('time').sort_index()


def get_spike_indicator_dataframe(neuron_key, animals,
                                  time_function=get_trial_time):
    ''' Returns a dataframe with a spike time indicator column
    where 1 indicates a spike at that time and 0 indicates no
    spike at that time. The number of datapoints corresponds
    is the same as the LFP.
    '''
    time = time_function(neuron_key, animals)
    spikes_df = get_spikes_dataframe(neuron_key, animals)
    time_index = np.digitize(spikes_df.index, time)
    return (spikes_df.groupby(time[time_index]).sum()
            .reindex(index=time, fill_value=0))


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
    try:
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
    except AttributeError:
        logger.warn('{0}, {1}, {2} not processed'.format(animal, day, epoch))


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
    return (dataframe.animal + '_' +
            dataframe.day.map('{:02d}'.format) + '_' +
            dataframe.epoch.map('{:02}'.format) + '_' +
            dataframe.tetrode_number.map('{:03}'.format) + '_' +
            dataframe.neuron_number.map('{:03}'.format))


def _add_to_dict(dictionary, tetrode_ind, neuron_ind):
    dictionary['tetrode_number'] = tetrode_ind + 1
    dictionary['neuron_number'] = neuron_ind + 1
    return dictionary
