import os
from os import listdir
from src.ml_logic.preproc import images_to_dataset

def create_dict_img(ENVPATH):
    '''
    creates two dictionnaries :

    dict_img
        keys = image IDs
        values = category

    dict_tnsr
        keys = image IDs
        values = tensor

    ARGS: 'ENVPATH' : Environment variable leading to images folder path

    returns: dict
    '''
    #instantiate lists and get basic path to images folder
    images = os.environ.get(ENVPATH)
    dict_img = {}
    img_list = []

    #loop over folders
    for cat in listdir(images)[1:]:
        #loop over files
        for img in listdir(f'{images}/{cat}'):
            img_list.append(img)
            dict_img[img.removesuffix('.jpg')] = cat #create dict and remove ".jpg"

    return dict_img




def create_dict_tnsr(ENVPATH):
    '''
    create dictionnary:
    keys = image IDs
    values = tensor

    ARGS: 'ENVPATH' : Environment variable leading to images folder path

    returns: dict
    '''
    images_tnsr = list(images_to_dataset(ENVPATH,
                                    validation_split=False)\
                                        .as_numpy_iterator())
    images = os.environ.get(ENVPATH)
    tnsr_list = []
    img_list = []
    for batch in images_tnsr:
        for tnsr in batch[0]:
            tnsr_list.append(tnsr)

    for cat in listdir(images)[1:]:
        for img in listdir(f'{images}/{cat}'):
            img_list.append(img.removesuffix('.jpg'))

    return dict(zip(img_list, tnsr_list))




def target_cat():
    '''
    Fonction pour faciliter la transformation de la data en trois catégories
    Args: None
    Returns: dictionnary
    '''
    classes = {'MEL': 'danger', 'NV':'benign',
                'BCC':'consult', 'AK' : 'consult',
                'BKL' : 'benign', 'DF' : 'benign',
                'VASC' : 'benign', 'SCC' : 'danger'}
    return classes
