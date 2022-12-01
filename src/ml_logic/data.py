import os
import shutil


def create_subset():

    dir_list = os.listdir(os.environ.get('IMAGE_DATA_PATH'))[1:]
    source_path = os.environ.get('IMAGE_DATA_PATH')
    subset_ = '../subset'
    subset_path = os.path.join(source_path, subset_)
    os.mkdir(subset_path)
    subset_size = int(os.environ.get('SUBSET_SIZE'))

    #iterate over source directories
    for dir in dir_list:
        dir_path = os.environ.get('IMAGE_DATA_PATH')
        #Create subset directories
        path = os.path.join(subset_path, dir)
        os.mkdir(path)
        for file_name in os.listdir(dir_path + '/' + dir)[:subset_size]:
            # print(f'{subset_path}/{folder}/{file_name}')
            shutil.copy(f'{source_path}/{dir}/{file_name}',
                        f'{subset_path}/{dir}/{file_name}',follow_symlinks=True)
