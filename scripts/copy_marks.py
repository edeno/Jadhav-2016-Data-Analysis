import re
import sys
from os import walk
from os.path import join
from shutil import copyfile


def main():
    src_directory = '/Volumes/Seagate Expansion Drive/misc/HPData/'
    target_directory = '/Users/edeno/Documents/GitHub/Jadhav-2016-Data-Analysis/Raw-Data/'

    mark_files = [join(root, f) for root, _, files in walk(src_directory)
                  for f in files if f.endswith('_params.mat')
                  and not f.startswith('matclust')]
    new_filenames = [rename_mark_file(mark_file, target_directory)
                     for mark_file in mark_files]

    for mark_file, new_filename in zip(mark_files, new_filenames):
        print('Copying {mark_file}\nto \n{new_filename}\n'.format(
            mark_file=mark_file,
            new_filename=new_filename
        ))
        copyfile(mark_file, new_filename)


def rename_mark_file(file_str, target_directory):
    animal = file_str.split('/')[-4]
    matched = re.match(
        r'(\d+)_.*-(\d+)_params.mat', file_str.split('/')[-1])
    try:
        day, tetrode_number = matched.groups()
    except AttributeError:
        matched = re.match(
            r'(\d+)-.*-(\d+)_params.mat', file_str.split('/')[-1])
        try:
            day, tetrode_number = matched.groups()
        except AttributeError:
            print(file_str)
            raise

    new_name = '{animal}marks{day}-{tetrode_number}.mat'.format(
        animal=animal,
        day=day,
        tetrode_number=tetrode_number
    )
    animal_directory = '{animal}_direct'.format(
        animal=animal)

    return join(target_directory, animal_directory, 'EEG', new_name)


if __name__ == '__main__':
    sys.exit(main())
