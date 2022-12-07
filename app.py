import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from src.ml_logic.model import load_Model_G
import os
import cv2
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input
import pickle
from src.ml_logic.registry import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims


SIDEBAR = st.sidebar.image("/home/naimabechari/code/naimabe/melanoma-project/data/lewagonimage.png", use_column_width=True)
st.sidebar.write("Batch #1061 - Data Science")
st.sidebar.write("## Who are we?")
st.sidebar.write("George, Dejan and Naïma")
st.sidebar.write("The Goal of our project is to predict whether a skin lesion is benign or malignant.")

st.markdown("<h1 style='text-align:center; color:black;'>PREDICTION OF MELANOMAS</h1", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:black;'>Is my Skin lesion a benign or malignant lesion?</h2", unsafe_allow_html=True)

uploaded_image = st.file_uploader(label = "You can drop your skin image here:", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


#model = load_Model_G()

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.success("Your image is downloaded !")

    st.image(image, width=400, use_column_width=False, caption='Your Picture')
    st.write("")
    #st.write("Classifying...")
    #image.save('test_image.jpeg')
    image_resized = image.resize((64,64))
    #image_resized = preprocess_input(image)
    array = img_to_array(image_resized)
    array = expand_dims(array, axis=0, name=None)

    if st.button("Predict"):
        """Model is loading..."""
        loaded_model = load_model()

        """Prediction of the image..."""
        prediction = loaded_model.predict(array)
        """Results: """
        st.write("Benign lesion:", round(prediction[0][0]*100, 2), "%", color='green')
        st.write("You need to consult:", round(prediction[0][1]*100, 2), "%" )
        st.write("Malignant lesion:", round(prediction[0][2]*100, 2), "%", color='red')
        if prediction.max() == prediction[0][0]:
            st.success("You don't need to worry")
        elif prediction.max() == prediction [0][1]:
            st.error("You need to see your doctor to be sure")
        else:
            st.error("Your lesion is dangerous")
    else:
        pass
