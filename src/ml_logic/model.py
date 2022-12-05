from keras.applications import EfficientNetB7
from keras.models import Sequential, Model
from tensorflow.keras.models import Sequential
from src.ml_logic.preproc import get_X_y
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from keras import layers
import tensorflow as tf



def initialise_EfficientNet_model():
    '''
    Initialises non-trainable basemodel EfficientNet, adds two layers and compiles for multicategorical classification.
    Args: None
    returns: Model
    '''

    base_model = EfficientNetB7(include_top=False,
                            weights='imagenet',
                            input_shape=(256, 256, 3),
                            trainable=False)
    #base_model.trainable = False
    base_model.add(layers.Flatten())
    base_model.add(layers.Conv2D(64, activation='relu'))
    base_model.add(layers.Dense(9, activation='softmax'))
    model = base_model.compile(loss='categorical_crossentropy',
                               optimizer=tf.keras.optimizers.Adam,
                               metrics=['accuracy', 'precision'])
    return model



# def initialise_WagNet():
#     model = Sequential()

def initialize_tabulaire_model():
    model = Sequential()
    model.add(layers.Dense(9, activation='softmax', input_dim=11))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
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
