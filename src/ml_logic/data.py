import os
import shutil


def create_subset():
    '''
    Fonction qui créé un directory "subset" dans le directory "data",
    contenant un directory pour chaque target category avec un nombre
    limité d'images en fonction de la SUBSET_SIZE,
    qui sera définie par votre variable d'environnement.
    '''
    source_path = os.environ.get('IMAGE_DATA_PATH')
    dir_list = os.listdir(source_path)
    subset_ = '../subset'
    subset_path = os.path.join(source_path, subset_)
    os.mkdir(subset_path)
    subset_size = int(os.environ.get('SUBSET_SIZE'))

    #iterate over source directories
    for dir in dir_list:
        #Create subset directories
        path = os.path.join(subset_path, dir)
        os.mkdir(path)
        for file_name in os.listdir(source_path + '/' + dir)[:subset_size]:
            #Copy files into new directories
            shutil.copy(f'{source_path}/{dir}/{file_name}',
                        f'{subset_path}/{dir}/{file_name}',follow_symlinks=True)
