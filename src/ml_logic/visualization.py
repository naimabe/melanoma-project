import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.ml_logic.preproc import images_to_dataset


def description_target():
    '''
    Function that shows the description of the target dataset (percentage of each category, their means, medians...)
    '''
    target_dataset = pd.read_csv('../data/ISIC_2019_Training_GroundTruth.csv')
    return target_dataset.describe()

def visualization_barplot_target():
    '''
    Function that shows the distribution of the category of skin lesion
    '''
    target_dataset = pd.read_csv('../data/ISIC_2019_Training_GroundTruth.csv')
    target_dataset = target_dataset.set_index('image')
    target_dataset.columns = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion', 'Squamous cell carcinoma', 'None']
    target_dataset = target_dataset.idxmax(axis='columns')
    target_dataset = target_dataset.value_counts()
    target_dataset = target_dataset.reset_index()
    target_dataset.columns = ['Melanoma type', 'count']
    # plot the distribution of the skin lesion
    barplot = sns.barplot(target_dataset, x='Melanoma type', y='count')
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation = 45, size=10, horizontalalignment='right');
    return barplot

def visualization_pie_target():
    '''
    Function that returns a pie with the percentage of each category of skin lesion
    '''
    target_dataset = pd.read_csv('../data/ISIC_2019_Training_GroundTruth.csv')
    target_dataset = target_dataset.set_index('image')
    target_dataset = target_dataset.idxmax(axis='columns')
    target_dataset = target_dataset.value_counts()
    target_dataset = target_dataset.reset_index()
    target_dataset.columns = ['Melanoma type', 'count']
    target_dataset['percentage'] = target_dataset['count']/target_dataset['count'].sum()*100
    target_dataset.round(decimals=2)
    labels = target_dataset['Melanoma type']
    sizes = target_dataset.percentage
    plt.figure(figsize=(15, 15))
    pieplot = plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, labeldistance=1.05, pctdistance=0.7);
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
