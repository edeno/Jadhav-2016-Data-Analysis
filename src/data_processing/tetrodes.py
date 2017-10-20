from os.path import join

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .core import RAW_DATA_DIR, _convert_to_dict


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
    try:
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
    except KeyError:
        return pd.DataFrame(tetrode_dict_list)


def get_trial_time(epoch_or_tetrode_key, animals):
    try:
        animal, day, epoch, tetrode_number = epoch_or_tetrode_key[:4]
        lfp_df = get_LFP_dataframe(
            (animal, day, epoch, tetrode_number), animals)
    except ValueError:
        # no tetrode number provided
        tetrode_info = (
            make_tetrode_dataframe(animals)
            .loc[epoch_or_tetrode_key]
            .set_index(['animal', 'day', 'epoch', 'tetrode_number']))
        lfp_df = pd.concat(
            [get_LFP_dataframe(tetrode_key, animals)
             for tetrode_key in tetrode_info.index],
            axis=1)

    return lfp_df.index


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
