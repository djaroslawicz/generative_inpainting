import argparse
import os
from random import shuffle
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--flist_path', default='../training_data', type=str,
                    help='The path for flist file')

# make an flist file as defined on the gen inpainting github
def make_flist(file_list, flist_name):
    flist = open(flist_name, 'w')
    for file in file_list:
        flist.write(file + '\n')
    flist.close()


if __name__ =="__main__":
    args = parser.parse_args()
    composer_list = ['vivaldi', 'bach', 'brahms', 'handel', 'haydn', 'mozart', 'schubert', 'beethoven']
    data_types = ['holdout', 'split', 'train']
    files = []
    # get all files
    for composer in composer_list:
        # The assumption is that this script is run from same directory containing
        # reconstructed_spec directory
        source_dir = './reconstructed_spec_stream/' + composer + '/train/'
        files += glob.glob(source_dir + '*.png')
        # shuffle for training
        shuffle(files)
        # split into train and validation (5%)
        split_point = int(.95*len(files))
        train_files = files[:split_point]
        val_files = files[split_point:]
        # make train and validation file flists
    make_flist(train_files, args.flist_path + 'train_shuffled.flist')
    make_flist(val_files, args.flist_path + 'validation_static_view.flist')

