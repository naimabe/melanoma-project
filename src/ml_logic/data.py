import os
import shutil
from src.ml_logic.utils import create_dict_img
from src.ml_logic.preproc import preprocessing_X_tabulaire


def create_subset():
    '''
    Fonction qui créé un directory "subset" dans le directory "data",
    contenant un directory pour chaque target category avec un nombre
    limité d'images en fonction de la SUBSET_SIZE,
    qui sera définie par votre variable d'environnement.
    '''
    source_path = os.environ.get('IMAGE_DATA_PATH')
    dir_list = os.listdir(source_path)[1:]
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



def create_tab_subset():
    '''
    Creates a subset dataframe to match the image_subset

    '''
    subset_dict = create_dict_img('IMAGE_DATA_PATH')
    X_tab = preprocessing_X_tabulaire('METADATA_CSV_PATH')

    for x in subset_dict.keys():
        if x in X_tab.index:
            X_tab_subset = X_tab.loc[x]
        else:
            X_tab.drop(labels=x, axis='index')


    #X_tab_subset = X_tab.drop(labels=)
    X_tab_subset
