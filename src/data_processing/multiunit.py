import numpy as np
import pandas as pd
from scipy.io import loadmat
from os.path import join
from .core import RAW_DATA_DIR
from .tetrodes import get_trial_time


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


def get_mark_indicator_dataframe(tetrode_key, animals,
                                 time_function=get_trial_time):
    time = time_function(tetrode_key[:3], animals)
    mark_dataframe = (get_mark_dataframe(tetrode_key, animals)
                      .loc[time.min():time.max()])
    time_index = np.digitize(mark_dataframe.index, time)
    return (mark_dataframe.groupby(time[time_index]).mean()
            .reindex(index=time))
