import pandas as pd
import numpy as np
from scipy.io import loadmat
from glob import glob
from os.path import join

from .core import RAW_DATA_DIR


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
