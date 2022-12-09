
import os
import shutil

from os import listdir
import albumentations as A
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def modify_target_csv():
    '''
    Prepares data for the move_images_tertiaire function
    '''
    df = pd.read_csv(os.environ.get('TARGET_CSV_PATH'))
    df = df.set_index('image')
    df = df.rename(columns={'MEL' : 'danger', 'BCC' : 'consult', 'DF' : 'benign'})
    df['benign'] = df['benign'] + df['NV'] + df['UNK'] + df['VASC'] + df['BKL']
    df['danger'] = df['danger'] + df['SCC']
    df['consult'] = df['consult'] + df['AK']
    df = df.drop(columns=['NV', 'AK', 'BKL', 'VASC', 'SCC', 'UNK'], axis=0)
    return df

def move_images_tertiaire():
    '''
    Moves images from training set into 3 different folders, named according to the target category of each image.
    Args: None
    Returns: None
    '''
    #Create metadata dataframe
    X_tab = preprocessing_X_tabulaire('METADATA_CSV_PATH')

    #Modify Target CSV according to target categories
    df = modify_target_csv()

    #Prepare source path
    source_path = os.environ.get('ORIGINAL_IMAGE_PATH')
    dir_list = ['danger', 'benign', 'consult']
    # images_ = '../' + 'images'
    image_path = '../data/images'
    os.mkdir(image_path)

    #iterate over source directories
    for dir in dir_list:
        #Create subset directories
        path = os.path.join(image_path, dir)
        os.mkdir(path)
        #copy files
        for file_name in os.listdir(source_path):
            if file_name.removesuffix('.jpg') in X_tab.index:
                if file_name.endswith('.jpg'):
                    if df.loc[file_name.removesuffix('.jpg')][dir] == 1:
                        #Copy files into new directories
                        shutil.copy(f'{source_path}/{file_name}',
                                f'{image_path}/{dir}/{file_name}',follow_symlinks=True)
            else:
                pass

def augmentation_pipeline(img):

    '''
    Augments image data by random cropping, horizontal flipping and changing the brightness contrast.

    Args: img (numpy array)

    returns: image (numpy array)
    '''

    img = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),])
    return img['image']


def normalise(img):
    '''
    Normalises images to values between 0 and 1.

    Args: img (numpy array)

    returns: img (numpy array)
    '''

    norm_img = np.zeros((256,256))
    normalised_img = cv.normalize(img,  norm_img, 0.00, 1.00)
    return normalised_img


def balance_data(X):
    '''
    Function that balances the dataset by adding weights to the under-represented classes.

    '''
    sampling_amount = 4522

    X_balanced = SMOTE(X, sampling_strategy={'MEL':sampling_amount,
                                'NV':sampling_amount,
                                'BCC':sampling_amount,
                                'AK' : sampling_amount,
                                'BKL' : sampling_amount,
                                'DF' : sampling_amount,
                                'VASC' : sampling_amount,
                                'SCC' : sampling_amount,
                                'UNK' : sampling_amount},
                                k_neighborsint=5,
                                n_jobs=-1)
    return X_balanced


def images_to_dataset(ENVPATH, validation_split=True, image_size=(64, 64)):
    '''
    Function that sort and transform images into a tensorflow dataset according to their classes

    Returns: Tensor (but should return Numpy or Dataframe)
    '''

    directory = os.environ.get(f'{ENVPATH}')
    if validation_split:
        dataset = image_dataset_from_directory(
                                    directory,
                                    labels='inferred',
                                    label_mode='int',
                                    class_names=None,
                                    color_mode='rgb',
                                    batch_size=32,
                                    image_size=image_size,
                                    shuffle=True,
                                    seed=123,
                                    validation_split=0.25,
                                    subset='training',
                                    follow_links=False,
                                    crop_to_aspect_ratio=False,
                                )
        dataset_val = image_dataset_from_directory(
                                    directory,
                                    labels='inferred',
                                    label_mode='int',
                                    class_names=None,
                                    color_mode='rgb',
                                    batch_size=32,
                                    image_size=image_size,
                                    shuffle=True,
                                    seed=123,
                                    validation_split=0.25,
                                    subset='validation',
                                    follow_links=False,
                                    crop_to_aspect_ratio=False,
                                )
        return dataset, dataset_val
    else:
        dataset = image_dataset_from_directory(
                                    directory,
                                    labels='inferred',
                                    label_mode='int',
                                    class_names=None,
                                    color_mode='rgb',
                                    batch_size=32,
                                    image_size=image_size,
                                    shuffle=False,
                                    seed=None,
                                    validation_split=None,
                                    subset= None,
                                    follow_links=False,
                                    crop_to_aspect_ratio=False,
                                )
        return dataset



def preprocessing_X_tabulaire(ENVPATH):

    """
    Cette function fait preprocessing des données tabulaires
    Args: X

    return: X_preprocessed

    """

    #load data
    df = pd.read_csv(os.environ.get(ENVPATH))

    #drop NaN and colummn 'lesion'

    df = df.dropna(axis=0, how='all', subset=['age_approx', 'anatom_site_general', 'sex'])

    #Drop colonne Lesion_id
    df = df.drop(['lesion_id'], axis=1)

    #replace NaN per "Delete*"
    df.sex.replace(np.nan, "Delete", inplace=True)
    df.anatom_site_general.replace(np.nan, "Delete1", inplace=True)

    #replace NaN per "mean" in column "age_approx"

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df[['age_approx']])
    df['age_approx'] = imputer.transform(df[['age_approx']])

    #transformation "string" to "numerique" in colummn "sex"
    #making news columns
    ohe = OneHotEncoder(sparse = False, handle_unknown='ignore')
    ohe.fit(df[['sex']])
    sex_encoded = ohe.transform(df[['sex']])
    df[ohe.categories_[0]] = sex_encoded

    #transformation "string" to "numerique" in colummn "anatom_site_general_encoded"
    #making news columns

    ohe2 = OneHotEncoder(sparse = False, handle_unknown='ignore')
    ohe2.fit(df[['anatom_site_general']])
    anatom_site_general_encoded = ohe2.transform(df[['anatom_site_general']])
    df[ohe2.categories_[0]] = anatom_site_general_encoded


    #transformation colummn "image" to Index
    X_preprocessed = df.set_index('image', inplace = True)

    #drop useless colummns
    X_preprocessed = df.drop(columns=['anatom_site_general', 'sex', 'Delete', 'Delete1'])

    #StandardScaler X data and transformation to DataFrame
    s_scaler = StandardScaler()
    X_preprocessed[:] = s_scaler.fit_transform(X_preprocessed)

    return X_preprocessed


def get_X_y():
    '''
    Cette fonction lit les deux tableaux .csv et sort un X_Preprocessed et un y
    '''
    y_df = pd.read_csv('../data/archive/ISIC_2019_Training_GroundTruth.csv')
    X_preproc = preprocessing_X_tabulaire()
    y_df = y_df.set_index('image')
    X_y = X_preproc.merge(y_df, how='left', on='image')
    target = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    X = X_y.drop(target, axis = 1)
    y = X_y[target]
    return X, y


def preprocessing_pipeline(ENVPATH, jumpfile=1):
    '''
    One function that loads tabular metadata, target information and images, turns all three into tensors and lines them up, thus preparing them for the concatenated model to receive as training data.

    ARGS:
        - ENVPATH: environment variable path leading to the image data
        - jumpfile: if the function returns an error it is likely that there is a hidden file in the image directory, jumpfile=1 will skip that file. If there is no hidden file, then set jumpfile=0 so as to avoid it missing some of your data.

    Returns:
        - img_input : a numpy array of all image tensors
        - tab_input : a numpy array of all tabular data
        - target_input : a numpy array of the target data
    '''

    #load tabular_data
    if ENVPATH == 'SUBSET_DATA_PATH':
        X_tab = create_tab_subset()
    else:
        X_tab = preprocessing_X_tabulaire('METADATA_CSV_PATH')

    #create image dictionary adapted to the data
    img_dict = create_dict_img(ENVPATH, jumpfile=jumpfile)

    #Create tensor dictionnary
    tnsr_dict = create_dict_tnsr(ENVPATH)

    #Create Target dictionary
    target_dict = {'benign' : 0.0,
                    'consult' : 1.0,
                    'danger' : 2.0}

    #add tensor and target columns to tabular data to keep all data sorted
    X_tab['img_tnsr'] = X_tab.index.map(tnsr_dict)
    X_tab['target'] = X_tab.index.map(img_dict)

    #Tabular data as tensor
    tab_input = tf.convert_to_tensor(X_tab.drop(columns=['img_tnsr', 'target']))

    #target data as tensor
    target_input = tf.convert_to_tensor(X_tab.target.map(target_dict))

    #Image data as tensor
    tensor_list = X_tab.img_tnsr.tolist()
    tnsr_list = [x / 255 for x in tensor_list]
    img_input = tf.convert_to_tensor(tnsr_list)

    return list(img_input), list(tab_input), list(target_input)


def create_dict_img(ENVPATH,jumpfile=0):
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
    for cat in listdir(images)[jumpfile:]:
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


def create_tab_subset(jumpfile=1):
    '''
    Creates a subset dataframe to match the image_subset

    '''
    subset_dict = create_dict_img('SUBSET_DATA_PATH',jumpfile=jumpfile)
    X_tab = preprocessing_X_tabulaire('METADATA_CSV_PATH')

    keys = list(subset_dict.keys())
    subset_tab = X_tab.loc[keys]
    return subset_tab





def preproc_train_test_split(img, tab, target):
    '''
    Train test split of image data, tab data and target data.
    '''
    img_train, img_test, target_train, target_test = train_test_split(
        img,
        target,
        test_size=0.25,
        random_state=1
    )

    tab_train, tab_test , _ , _ = train_test_split(
        tab,
        target,
        test_size=0.25,
        random_state=1
        )
    elements = [img_train, img_test, target_train, target_test, tab_train, tab_test]

    tensors = [tf.convert_to_tensor(element) for element in elements]

    img_train = tensors[0]
    img_test = tensors[1]
    tab_train = tensors[4]
    tab_test = tensors[5]
    target_train = tensors[2]
    target_test = tensors[3]

    return img_train, img_test, tab_train, tab_test, target_train, target_test
