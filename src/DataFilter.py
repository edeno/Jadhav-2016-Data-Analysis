import os
import sys
import scipy.io


def get_data_filename(animal, day, file_type):
    '''
    Returns the Matlab file name assuming it is in the Raw Data directory.
    File type is a string that refers to the various data structure names
    (DIO, tasks, linpos) Animal is a named tuple. Day is an integer giving the
    recording session day.
    '''
    data_dir = '{working_dir}/Raw-Data/'.format(working_dir=os.getcwd())
    return '{data_dir}/{animal.directory}/{animal.short_name}{file_type}{day:02d}.mat'.format(
        data_dir=data_dir,
        animal=animal,
        file_type=file_type,
        day=day)


def get_epoch(animal, days, epoch_type='', environment=''):
    '''
    Returns a list of two-element tuples (day, epoch index) that
    can access the data structure within each Matlab file. Epoch type
    is the task for that epoch (sleep, run, etc.) and environment is
    the type of maze the animal is in (if any). If no epoch type or
    environment is given, returns all epoch types and environments
    '''
    epoch_index = list()

    if isinstance(days, int):
        days = [days]

    for day in days:
        try:
            task_file = scipy.io.loadmat(get_data_filename(animal, day, 'task'))
            filtered_epochs = [(ind, epoch) for ind, epoch in enumerate(task_file['task'][0, day-1][0])
                               if epoch['type'] == epoch_type or
                               epoch_type == '']
            epoch_index += [(day, ind) for ind, epoch in filtered_epochs
                    if ('environment' in epoch.dtype.names and
                    epoch['environment'] == environment) or
                    environment == '']
        except IOError as err:
            print('I/O error({0}): {1}'.format(err.errno, err.strerror))
            sys.exit()

    return epoch_index


def get_var(animal, day, file_type, variable, epoch_type='', environment=''):
    '''
    Returns a filtered list containing the data structures corresponding to the
    animal, day, file_type, epoch specified.
    '''
    epoch = get_epoch(animal, day, epoch_type=epoch_type, environment=environment)
    file = scipy.io.loadmat(get_data_filename(animal, day, file_type))
    return [file[variable][0, day - 1][0, ind] for day, ind in epoch]


def get_DIO_var(animal, day, dio_var, epoch_type='', environment=''):
    '''
    Returns a list of lists given a DIO variable (pulsetimes, timesincelast,
    pulselength, and pulseind) with a length corresponding to the number of
    epochs (first level) and the number of active pins (second level)
    '''
    epoch_pins = get_var(animal, day, 'DIO', 'DIO', epoch_type=epoch_type, environment=environment)
    return [pin[0][dio_var]
            for pins in epoch_pins
            for pin in pins.swapaxes(0, 1)
            if pin[0].dtype.names is not None]


def get_pos_var(animal, day, pos_var, epoch_type='', environment=''):
    '''
    Returns a list of lists given a pos variable (time, x, y, dir, vel, x-sm,
    y-sm, dir-sm, and vel-sm) with a length corresponding to the number of
    epochs (first level)
    '''
    field_names = ['time', 'x', 'y', 'dir', 'vel', 'x-sm', 'y-sm', 'dir-sm', 'vel-sm']
    field_ind = [field_names.index(var) for var in pos_var]
    epoch_pos = get_var(animal, day, 'pos', 'pos', epoch_type=epoch_type, environment=environment)
    return [pos['data'][0, 0][:, field_ind]
            for pos in epoch_pos]
