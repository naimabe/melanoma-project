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
  - MEL = Melanoma (CANCER)
  - NV = Melanocytic nevus
  - BCC = Basal cell carcinoma (CANCER)
  - AK = Actinic keratosis
  - BKL = Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
  - DF = Dermatofibroma
  - VASC = Vascular lesion
  - SCC = Squamous cell carcinoma (CANCER)
  - UNK = None of the above


  - MEL = Melanoma: (CANCER)
        Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin — the pigment that gives your skin its color. Melanoma can also form in your eyes and, rarely, inside your body, such as in your nose or throat.

  - NV = Melanocytic nevus:
        a skin condition characterized by an abnormally dark, noncancerous skin patch (nevus) that is composed of pigment-producing cells called melanocytes. It is present from birth (congenital) or is noticeable soon after birth.

  - BCC = Basal cell carcinoma: (CANCER)
        A type of skin cancer that begins in the basal cells.
        Basal cells produce new skin cells as old ones die. Limiting sun exposure can help prevent these cells from becoming cancerous.
        This cancer typically appears as a white, waxy lump or a brown, scaly patch on sun-exposed areas, such as the face and neck.

  - AK = Actinic keratosis:
        Actinic keratoses (also called solar keratoses) are dry scaly patches of skin that have been damaged by the sun. The patches are not usually serious. But there's a small chance they could become skin cancer.

  - BKL = Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis):
        Benign keratosis is a common noncancerous skin growth. People tend to get more of them as they get older.

  - DF = Dermatofibroma:
        A cellular dermatofibroma is a noncancerous skin growth. It may look like a small, firm bump, similar to a mole. Unlike other dermatofibromas, cellular dermatofibromas often attach to your deepest layer of skin. Because they're noncancerous, they usually don't need treatment.

  - VASC = Vascular lesion:
        Vascular lesions are relatively common abnormalities of the skin and underlying tissues, more commonly known as birthmarks. There are three major categories of vascular lesions: Hemangiomas, Vascular Malformations, and Pyogenic Granulomas.


  - SCC = Squamous cell carcinoma: (CANCER)
        Squamous cell carcinoma (SCC) of the skin is the second most common form of skin cancer, characterized by abnormal, accelerated growth of squamous cells. When caught early, most SCCs are curable.


  - UNK = None of the above
        Unknown



## Project setup
## Environment

## First step : Preprocessing of the data
    Images:
      The images are classified in folders according to their classes and resized.

    Metadata:
