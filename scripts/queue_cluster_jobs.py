'''Script for executing run_by_epoch on the cluster
'''
from argparse import ArgumentParser
from os import getcwd, makedirs, environ
from os.path import join
from subprocess import run
from sys import exit

from loren_frank_data_processing import make_epochs_dataframe
from src.parameters import ANIMALS


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--Animal', type=str, help='Short name of animal')
    parser.add_argument('--Day', type=int, help='Day of recording session')
    parser.add_argument('--Epoch', type=int,
                        help='Epoch number of recording session')
    return parser.parse_args()


def queue_job(python_cmd, directives=None, log_file='log.log',
              job_name='job'):
    queue_cmd = (
        'qsub {directives} -j y -o {log_file} -N {job_name}').format(
            directives=directives,
            log_file=log_file,
            job_name=job_name)
    cmd_line_script = ' | '.join([
        'echo python {python_cmd}'.format(python_cmd=python_cmd),
        queue_cmd])
    run(cmd_line_script, shell=True)


def main():
    # Set the maximum number of threads for openBLAS to use.
    NUM_THREADS = 16
    environ['OPENBLAS_NUM_THREADS'] = str(NUM_THREADS)
    environ['NUMBA_NUM_THREADS'] = str(NUM_THREADS)
    environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
    log_directory = join(getcwd(), 'logs')
    makedirs(log_directory,  exist_ok=True)

    python_function = 'run_by_epoch.py'
    directives = ' '.join(
        ['-l h_rt=2:00:00', '-pe omp {0}'.format(NUM_THREADS),
         '-P braincom', '-notify', '-l mem_total=125G',
         '-v OPENBLAS_NUM_THREADS', '-v NUMBA_NUM_THREADS',
         '-v OMP_NUM_THREADS'])

    args = get_command_line_arguments()
    if args.Animal is None and args.Day is None and args.Epoch is None:
        epoch_info = make_epochs_dataframe(ANIMALS)
        epoch_keys = epoch_info[(epoch_info.type == 'run') & (
            epoch_info.environment != 'lin')].index
    else:
        epoch_keys = [(args.Animal, args.Day, args.Epoch)]

    for (animal, day, epoch_ind) in epoch_keys:
        print('Animal: {0}, Day: {1}, Epoch: {2}'.format(
            animal, day, epoch_ind))
        log_file = '{animal}_{day:02d}_{epoch:02d}.log'.format(
            animal=animal, day=day, epoch=epoch_ind)
        job_name = (
            '{function_name}_{animal}_{day:02d}_{epoch:02d}').format(
            animal=animal, day=day, epoch=epoch_ind,
            function_name=python_function.replace('.py', ''))
        python_cmd = '{python_function} {animal} {day} {epoch}'.format(
            python_function=python_function, animal=animal, day=day,
            epoch=epoch_ind)
        queue_job(python_cmd,
                  directives=directives,
                  log_file=join(log_directory, log_file),
                  job_name=job_name)


if __name__ == '__main__':
    exit(main())
