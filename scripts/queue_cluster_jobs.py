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
    os.makedirs(log_directory,  exist_ok=True)

    python_function = 'run_by_epoch.py'
    directives = '-l h_rt=10:00:00 -pe omp 8 -P braincom'

    Animal = collections.namedtuple('Animal', {'directory', 'short_name'})
    num_days = 8
    days = range(1, num_days + 1)
    animals = {'HPa': Animal(directory='HPa_direct', short_name='HPa'),
               'HPb': Animal(directory='HPb_direct', short_name='HPb'),
               'HPc': Animal(directory='HPc_direct', short_name='HPc')
               }
    epoch_info = data_processing.make_epochs_dataframe(animals, days)
    epoch_index = epoch_info[(epoch_info.type == 'run') & (epoch_info.environment != 'lin')].index

    for epoch in epoch_index:
        print(epoch)
        animal, day, epoch_ind = epoch
        log_file = '{animal}_{day:02d}_{epoch:02d}.log'.format(
            animal=animal, day=day, epoch=epoch_ind)
        function_name = '{function_name}_{animal}_{day:02d}_{epoch:02d}'.format(
            animal=animal, day=day, epoch=epoch_ind,
            function_name=python_function.replace('.py', ''))
        python_cmd = 'echo python {python_function} {animal} {day} {epoch}'.format(
            python_function=python_function,
            animal=animal,
            day=day,
            epoch=epoch_ind)
        queue_cmd = 'qsub {directives} -j y -o {log_file} -N {function_name}'.format(
            directives=directives,
            log_file=os.path.join(log_directory, log_file),
            function_name=function_name)
        script = ' | '.join([python_cmd, queue_cmd])
        subprocess.run(script, shell=True)


if __name__ == '__main__':
    sys.exit(main())
