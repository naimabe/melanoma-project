
import os
import shutil
from pathlib import Path

import albumentations as A
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from imblearn.over_sampling import SMOTE
from PIL import Image
from keras.utils import image_dataset_from_directory, to_categorical

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


def images_to_dataset(ENVPATH, validation_split=True):
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
                                    image_size=(64, 64),
                                    shuffle=True,
                                    seed=123,
                                    validation_split=0.3,
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
                                    image_size=(64, 64),
                                    shuffle=True,
                                    seed=123,
                                    validation_split=0.3,
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
                                    image_size=(64, 64),
                                    shuffle=False,
                                    seed=None,
                                    validation_split=None,
                                    subset= None,
                                    follow_links=False,
                                    crop_to_aspect_ratio=False,
                                )
        return dataset


def get_X_y():
    '''
    Cette fonction lit les deux tableaux .csv et sort un X_Preprocessed et un y
    '''
    y = pd.read_csv(os.environ.get('TARGET_CSV_PATH'))
    X_preprocessed = preprocessing_X_tabulaire(X)
    df = X_preprocessed.merge(y, on='image', how='inner')
    X = df.drop(['target']) # à corriger en fonction de la fonction preprocessing
    y = df.target # à corriger en fonction de la fonction preprocessing
    return X, y


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


#def create_small_test_dataset(sample_size):
