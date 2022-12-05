import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

SIDEBAR = st.sidebar.image("/home/naimabechari/code/naimabe/melanoma-project/data/lewagonimage.png", use_column_width=True)
st.sidebar.write("Batch #1061 - Data Science")
st.sidebar.write("## Who are we?")
st.sidebar.write("George, Dejan et Naïma")
st.sidebar.write("The Goal of our project is to predict whether a skin lesion is benign or malignant.")


st.markdown("""# PREDICTION OF MELANOMAS
## Is my Skin lesion a benign or malign lesion?""")


uploaded_image = st.file_uploader("You can drop your skin image here:")

if uploaded_image is not None:
    st.image(uploaded_image, caption='Your Picture')





#image = Image.open('sunrise.jpg')

#st.image(image, caption='Sunrise by the mountains')
