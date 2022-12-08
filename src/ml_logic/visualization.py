import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.ml_logic.preproc import images_to_dataset
import os

def description_target():
    '''
    Function that shows the description of the target dataset (percentage of each category, their means, medians...)
    '''
    target_dataset = pd.read_csv('../data/ISIC_2019_Training_GroundTruth.csv')
    return target_dataset.describe()

def dataset_creation_categories():
    '''
    Function that creates a dataset with the number of each category of melanoma
    '''
    target_dataset = pd.read_csv(os.environ.get('GROUNDTRUTH_PATH'))
    target_dataset = target_dataset.set_index('image')
    target_dataset.columns = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion', 'Squamous cell carcinoma', 'None']
    target_dataset = target_dataset.idxmax(axis='columns')
    target_dataset = target_dataset.value_counts()
    target_dataset = target_dataset.reset_index()
    target_dataset.columns = ['Melanoma type', 'count']
    return target_dataset

def visualization_barplot_target():
    '''
    Function that shows the distribution of the category of skin lesion
    '''
    plt.figure(figsize=(15,8))
    data = dataset_creation_categories()
    # plot the distribution of the skin lesion
    barplot = sns.barplot(data=data, x='Melanoma type', y='count').set(title='Distribution of Melanoma types')
    #barplot.set_xticklabels(barplot.get_xticklabels(), rotation = 45, size=10, horizontalalignment='right');
    return barplot

def visualization_pie_target():
    '''
    Function that returns a pie with the percentage of each category of skin lesion
    '''
    target_dataset = dataset_creation_categories()
    target_dataset['percentage'] = target_dataset['count']/target_dataset['count'].sum()*100
    target_dataset.round(decimals=2)
    labels = target_dataset['Melanoma type']
    sizes = target_dataset.percentage
    plt.figure(figsize=(15, 15))
    pieplot = plt.pie(sizes, labels=labels, autopct='%5.5f%%', shadow=False, labeldistance=1.05, pctdistance=0.7);
    return pieplot

def visualization_images():
    '''
    Function that shows an example of each category of skin lesion
    '''
    dataset = images_to_dataset()
    class_names = dataset.class_names
    plt.figure(figsize=(15, 15))
    for images, labels in dataset.take(1):
        for i in range(8):
            ax = plt.subplot(4, 2, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def dataset_creation_dangerousness():
    '''
    Function that creates a dataset with the dangerousness of the target(dangerous, potentially dangerous or benign)
    '''
    df = pd.read_csv(os.environ.get'METADATA_PATH')
    target = pd.read_csv('../data/ISIC_2019_Training_GroundTruth.csv')
    target_dataset = target.set_index('image')
    target_dataset = target_dataset.idxmax(axis='columns')
    target_dataset = target_dataset.reset_index()
    target_dataset.columns = ['image', 'class']
    df = df.merge(target_dataset, how='outer', on='image')
    data = df.replace({'NV':'Benign', 'MEL':'Danger', 'BCC':'Consult', 'BKL':'Benign', 'AK':'Consult', 'SCC':'Danger', 'VASC':'Benign', 'DF':'Benign'})
    data.rename(columns={'class': 'Melanoma type'}, inplace=True)
    return data

def visualization_dangerousness():
    '''
    Function that plots the count of each category of melanoma according to their dangerousness
    '''
    data = dataset_creation_dangerousness()
    sns.countplot(data=data, x=data['Melanoma type'], order=['Benign', 'Consult', 'Danger'], palette=['#33a02c','#ff7f00', '#e31a1c']).set(title='Distribution of Benign, Dangerous and Potentially Dangerous Skin Lesion')


def visualization_ages_vs_dangerousness():
    '''
    Function that plots the count of melanoma according to the age and the dangerousness
    '''
    data = dataset_creation_dangerousness()
    plt.figure(figsize=(10,6))
    sns.histplot(x="age_approx", hue="Melanoma type", hue_order=['Benign', 'Consult', 'Danger'], palette=['#33a02c','#ff7f00', '#e31a1c'], data=data, kde=True, multiple="stack")
    plt.show()

def visualization_anatom_vs_dangerousness():
    '''
    Function that plots the count of melanoma according to the anatomic site and the dangerousness
    '''
    data = dataset_creation_dangerousness()
    plt.figure(figsize=(14,6))
    sns.countplot(x="anatom_site_general", hue="Melanoma type", hue_order=['Benign', 'Consult', 'Danger'], palette=['#33a02c','#ff7f00', '#e31a1c'], data=data)
    plt.legend(loc="upper right")
    plt.show()


def plot_history(history):
    plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.legend()
