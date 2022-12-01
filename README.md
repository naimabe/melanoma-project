# melanoma-project

## Objective

The Goal of this project is to classify what is the diagnostic of a skin lesion among 9 categories.

1. Preprocess the data
2. Run 2 models (one for the images and one for the tabular data)
3. Concatenate the 2 models
4. Find a final model
5. Put the model in production

## Context

We have two csv and one folder with all the images.
There are 9 categories of skin lesion.
We want to create a model that predict which category a skin lesion is thank to a photo of the skin lesion and with some other informations (sex, age...).

Description of each category :
  - MEL = Melanoma
  - NV = Melanocytic nevus
  - BCC = Basal cell carcinoma
  - AK = Actinic keratosis
  - BKL = Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
  - DF = Dermatofibroma
  - VASC = Vascular lesion
  - SCC = Squamous cell carcinoma
  - UNK = None of the above


## Project setup
## Environment

## First step : Preprocessing of the data
    Images:
      The images are classified in folders according to their classes and resized.

    Metadata:
