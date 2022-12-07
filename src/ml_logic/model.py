import tensorflow as tf
import keras

from keras import layers
from keras.applications import EfficientNetB0
from keras.models import  Sequential, Functional, Model
from keras.layers import concatenate
from src.ml_logic.preproc import images_to_dataset
from keras.applications.efficientnet import preprocess_input

def load_Model_simple(input_shape=(64, 64, 3)):

    def load_efficientnet(input_shape=input_shape):
        '''
        load pre-trained model
        Args: None
        return: pre-trained model
        '''
        input_img = keras.Input(shape=input_shape)
        model = EfficientNetB0(include_top=False,
                            weights='imagenet',
                            input_shape=input_shape)
        model(preprocess_input(input_img))
        return model

    def set_nontrainable_layers(model):
        '''
        Ensures pre-trained model is not trainable on new data.
        Args: None
        return: pre-trained model
        '''
        model.trainable = True
        return model

    def add_last_layers():
        '''
        Adds final layers to the pre-trained model.
        Args: None
        return: pre-trained model with final layers
        '''
        base_model = load_efficientnet()
        base_model = set_nontrainable_layers(base_model)
        flattening_layer = layers.Flatten()
        dense_layer_01 = layers.Dense(120, activation='relu')
        dropout_01 = layers.Dropout(0.8)
        dense_layer_02 = layers.Dense(32, activation='relu')
        dropout_02 = layers.Dropout(0.6)
        prediction_layer = layers.Dense(3, activation='softmax')

        model = Sequential([base_model,
                            flattening_layer,
                            dense_layer_01,
                            dropout_01,
                            dense_layer_02,
                            dropout_02,
                            prediction_layer])

        # model.add_metric('recall', recall_score())

        model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        metrics=['accuracy'])

        return model

    model = add_last_layers()
    return model


    '''
    Train model with earlystopping and batch size parameters
    Args:

    KwArgs:

    '''

def initialize_tabulaire_model():
    model = Sequential()
    model.add(layers.Dense(9, activation='softmax', input_dim=11))
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model

def model_concat():
    '''
    Cette fonction fait la concatenation du modèle tabulaire avec le modèle images.
    '''
    #Inputs
    input_img = keras.Input(shape=(64, 64, 3))
    input_tab = layers.Input(shape=(11,))

    #CNN EfficientNet model
    eff_model = EfficientNetB0(include_top=False,
                                weights='imagenet',
                                input_shape=(64, 64, 3))
    eff_model.trainable = True

    outeff = eff_model(preprocess_input(input_img))
    flattening_layer = layers.Flatten()(outeff)
    dense_layer_01 = layers.Dense(32, activation='relu')(flattening_layer)
    dropout_01 = layers.Dropout(0.6)(dense_layer_01)

    # Second model
    dense_02 = layers.Dense(200, activation='relu')(input_tab)
    dropout_02 = layers.Dropout(0.6)(dense_02)

    # Concatenate outputs
    concat_outputs = concatenate([dropout_01, dropout_02])

    #Prediction layer
    prediction_layer = layers.Dense(3, activation='softmax')(concat_outputs)

    model = Model(inputs=[input_img, input_tab], outputs=[prediction_layer])

    model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=keras.optimizers.Adam(learning_rate=0.001),
                            metrics=['accuracy'])
    return model

def train_model_concat(model, img, data, target, batch_size, epochs):
    history = model.fit(x=[img, data],
                        y=target,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.25,
                        verbose=0)
    return history

def train_model_simple(model, batch_size, epochs):

    dataset = images_to_dataset('IMAGE_DATA_PATH',
                                image_size=(64, 64),
                                validation_split=False)

    history = model.fit(x=dataset,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.25,
                        verbose=0)

    return history

def predict_simple(model, ENVPATH, filename, tab=None):
    '''
    predicts the classification of an image.
    '''
    img_path = ENVPATH + '/' + filename
    image = images_to_dataset(img_path, validation_split=False)
    if tab:
        pred = model.predict([image, tab])
    else:
        pred = model.predict(image)

    return pred

def model_concat_02():
    '''
    Cette fonction fait la concatenation du modèle tabulaire avec le modèle images.
    '''
    #Inputs
    input_img = keras.Input(shape=(64, 64, 3))
    input_tab = layers.Input(shape=(11,))

    #CNN EfficientNet model
    eff_model = EfficientNetB0(include_top=False,
                                weights='imagenet',
                                input_shape=(64, 64, 3))
    eff_model.trainable = True
    outeff = eff_model(preprocess_input(input_img))
    flattening_layer = layers.Flatten()(outeff)
    dense_layer_01 = layers.Dense(80, activation='relu')(flattening_layer)
    dropout_01 = layers.Dropout(0.7)(dense_layer_01)
    dense_layer_02 = layers.Dense(32, activation='relu')(dropout_01)
    dropout_02 = layers.Dropout(0.7)(dense_layer_02)

    # Second model
    dense_tab_02 = layers.Dense(80, activation='relu')(input_tab)
    dropout_tab_02 = layers.Dropout(0.6)(dense_tab_02)
    dense_tab_03 = layers.Dense(32, activation='relu')(dropout_tab_02)
    dropout_tab_03 = layers.Dropout(0.6)(dense_tab_03)

    # Concatenate outputs
    concat_outputs = concatenate([dropout_02, dropout_tab_03])

    #Prediction layer
    prediction_layer = layers.Dense(3, activation='softmax')(concat_outputs)

    model = Model(inputs=[input_img, input_tab], outputs=[prediction_layer])

    model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                            metrics=['accuracy'])
    return model
