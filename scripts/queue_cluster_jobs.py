'''Script for executing run_by_epoch on the cluster
'''
from collections import namedtuple
from os import getcwd, makedirs
from os.path import join
from subprocess import run
from sys import exit

from src.data_processing import make_epochs_dataframe


def main():
    log_directory = join(getcwd(), 'logs')
    makedirs(log_directory,  exist_ok=True)

    python_function = 'run_by_epoch.py'
    directives = ('-l h_rt=6:00:00 '
                  '-pe omp 12 '
                  '-P braincom '
                  '-notify '
                  '-l mem_per_core=3G')

    Animal = namedtuple('Animal', {'directory', 'short_name'})
    num_days = 8
    days = range(1, num_days + 1)
    animals = {'HPa': Animal(directory='HPa_direct', short_name='HPa'),
               'HPb': Animal(directory='HPb_direct', short_name='HPb'),
               'HPc': Animal(directory='HPc_direct', short_name='HPc')
               }
    epoch_info = make_epochs_dataframe(animals, days)
    epoch_index = epoch_info[(epoch_info.type == 'run') & (
        epoch_info.environment != 'lin')].index

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
            log_file=join(log_directory, log_file),
            function_name=function_name)
        script = ' | '.join([python_cmd, queue_cmd])
        run(script, shell=True)


if __name__ == '__main__':
    exit(main())
