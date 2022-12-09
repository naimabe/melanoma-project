import streamlit as st
from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input

from src.ml_logic.registry import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims

#SIDEBAR
# SIDEBAR = st.sidebar.image("/home/naimabechari/code/naimabe/melanoma-project/data/lewagonimage.png", use_column_width=True)
st.sidebar.write("Batch #1061 - Data Science")
st.sidebar.write("## Who are we?")
st.sidebar.write("George, Dejan and Naïma")
st.sidebar.write("The Goal of our project is to predict whether a skin lesion is benign or malignant.")

#PRINCIPAL
st.markdown("<h1 style='text-align:center; color:black;'>PREDICTION OF MELANOMAS</h1", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:black;'>Is my Skin lesion a benign or malignant lesion?</h2", unsafe_allow_html=True)

st.write('First, we need some informations about you:')
#Selection of the features:
option_sex = st.selectbox(
    'Select your biological gender',
    ('Male', 'Female'))
option_age = st.slider('How old are you?', 0, 100, 25)
option_anatom = st.selectbox(
    'Where is located your lesion?',
    ('anterior torso', 'upper extremity', 'posterior torso', 'head/neck', 'lower extremity', 'palms/soles', 'oral/genital', 'lateral torso'))

st.write('You are a', option_age, 'years old', option_sex,'','.')
st.write('Your lesion is located in the', option_anatom, 'area.')


#Uploading of the image
uploaded_image = st.file_uploader(label = "You can now drop your skin image here:", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.success("Your image is downloaded !")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('')
    with col2:
        st.image(image, use_column_width=True, caption='Your Picture')
    with col3:
        st.write(' ')

    st.write("")
    #st.write("Classifying...")
    image_resized = image.resize((64,64))
    image_resized = preprocess_input(image_resized)
    array = img_to_array(image_resized)
    array = expand_dims(array, axis=0, name=None)

    if st.button("Predict"):
        loaded_model = load_model(path='models1 test')
        """The Model is loaded!"""
        """Your image is analyzed..."""
        prediction = loaded_model.predict(array)
        elif prediction.max() == prediction [0][1]:
            st.error("You need to see your doctor to be sure..")
        else:
            st.error("Your lesion is dangerous.")
    else:
        pass
