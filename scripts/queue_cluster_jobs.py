'''Script for executing run_by_epoch on the cluster
'''
from os import getcwd, makedirs
from os.path import join
from subprocess import run
from sys import exit

from src.data_processing import make_epochs_dataframe
from src.parameters import ANIMALS, N_DAYS


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
    log_directory = join(getcwd(), 'logs')
    makedirs(log_directory,  exist_ok=True)

    python_function = 'run_by_epoch.py'
    directives = ' '.join(['-l h_rt=6:00:00', '-pe omp 12', '-P braincom',
                           '-notify', '-l mem_per_core=3G'])

    epoch_info = make_epochs_dataframe(ANIMALS, range(1, N_DAYS + 1))
    epoch_keys = epoch_info[(epoch_info.type == 'run') & (
        epoch_info.environment != 'lin')].index

    for epoch_key in epoch_keys:
        print(epoch_key)
        animal, day, epoch_ind = epoch_key
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

    # Collect analysis after all epoch jobs have run
    analysis_directives = ' '.join(
        ['-pe omp 4', '-l h_rt=4:00:00', '-P braincom',
         '-l mem_per_core=2G',
         '-hold_jid "{epoch_function}*"'.format(
             epoch_function=python_function.replace('.py', ''))
         ])
    queue_job('collect_analysis.py',
              directives=analysis_directives,
              log_file=join(log_directory, 'collect_analysis.log'),
              job_name='collect_analysis')


if __name__ == '__main__':
    exit(main())
