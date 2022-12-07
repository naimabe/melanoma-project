import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping
from keras.models import  Sequential, Functional
from keras.layers import concatenate
from src.ml_logic.preproc import images_to_dataset

import keras

from keras.models import Model, Sequential
import os

def load_Model_simple(input_shape=(64, 64, 3)):

    def load_efficientnet(input_shape=input_shape):
        '''
        load pre-trained model
        Args: None
        return: pre-trained model
        '''
        model = EfficientNetB0(include_top=False,
                            weights='imagenet',
                            input_shape=input_shape)
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

def train_model(model, X, y, batch_size=64, patience=5, validation_split=0.3):
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
    outeff = eff_model(input_img)
    flattening_layer = layers.Flatten()(outeff)
    dense_layer_01 = layers.Dense(32, activation='relu')(flattening_layer)
    dropout_01 = layers.Dropout(0.5)(dense_layer_01)

    # Second model
    dense_02 = layers.Dense(5, activation='relu')(input_tab)

    # Concatenate outputs
    concat_outputs = concatenate([dropout_01, dense_02])

    #Prediction layer
    prediction_layer = layers.Dense(3, activation='softmax')(concat_outputs)

    model = Functional(inputs=[input_img, input_tab], outputs=[prediction_layer])

    model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                            metrics=['accuracy'])
    return model

def train_model(model):
    X, y = get_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    history = model.fit(X_train, y_train,
            batch_size=32,
            epochs=10,
            validation_split=0.3,
            verbose=0)
    return model,X_test,y_test

def predict(model, ENVPATH, filename, tab=None):
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
