#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import collections
import subprocess
sys.path.append(os.path.join(os.path.abspath(os.path.pardir), 'src'))
import data_processing


def main():
    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    python_function = 'run_by_epoch.py'
    directives = '-l h_rt=1:00:00 -pe omp 4'

    Animal = collections.namedtuple('Animal', {'directory', 'short_name'})
    num_days = 8
    days = range(1, num_days + 1)
    animals = {'HPa': Animal(directory='HPa_direct', short_name='HPa'),
               'HPc': Animal(directory='HPc_direct', short_name='HPc')
               }
    epoch_info = data_processing.make_epochs_dataframe(animals, days)
    epoch_index = (epoch_info
                   .loc[(['HPa', 'HPc'], [8]), :]
                   .loc[epoch_info.environment == 'wtr1'].index)
    for epoch in epoch_index:
        print(epoch)
        animal, day, epoch_ind = epoch
        log_file = '{animal}_{day}_{epoch}.txt'.format(
            animal=animal, day=day, epoch=epoch_ind, log_directory=log_directory)
        python_cmd = 'echo python {python_function} {animal} {day} {epoch}'.format(
            python_function=python_function,
            animal=animal,
            day=day,
            epoch=epoch_ind)
        queue_cmd = 'qsub {directives} -j y -o {log_file} -N {function_name}'.format(
            directives=directives,
            log_file=os.path.join(log_directory, log_file),
            function_name=python_function.replace('.py', ''))

        script = ' | '.join([python_cmd, queue_cmd])
        print(script)
        subprocess.run(script, shell=True)


if __name__ == '__main__':
    sys.exit(main())
