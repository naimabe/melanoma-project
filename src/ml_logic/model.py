import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential, Functional
import keras

from keras.models import Model, Sequential
import os

def load_Model_G():

    def load_efficientnet():
        '''
        load pre-trained model
        Args: None
        return: pre-trained model
        '''
        model = EfficientNetB0(include_top=False,
                            weights='imagenet',
                            input_shape=(64, 64, 3))
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
        dense_layer_01 = layers.Dense(32, activation='relu')
        dropout_01 = layers.Dropout(0.5)
        # dense_layer_02 = layers.Dense(30, activation='relu')
        # dropout_02 = layers.Dropout(0.5)
        prediction_layer = layers.Dense(3, activation='softmax')

        model = Sequential([base_model,
                            flattening_layer,
                            dense_layer_01,
                            dropout_01,
                            # dense_layer_02,
                            # dropout_02,
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
    input_1 = keras.Input(shape=(64, 64, 3))

    base_model = EfficientNetB0(include_top=False,
                                weights='imagenet',
                                input_shape=(64, 64, 3))
    base_model.trainable = True
    outeff = base_model(input_1)
    flattening_layer = layers.Flatten()(outeff)
    dense_layer_01 = layers.Dense(32, activation='relu')(flattening_layer)
    dropout_01 = layers.Dropout(0.5)(dense_layer_01)

    # The other input
    input_2 = layers.Input(shape=(11,))
    dense_2 = layers.Dense(5, activation='relu')(input_2)

    # Concatenate
    concat = layers.concatenate([dropout_01, dense_2])

    #Prediction layer
    prediction_layer = layers.Dense(3, activation='softmax')(concat)

    model = Functional(inputs=[input_1, input_2], outputs=[prediction_layer ])

    model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
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
