from keras.applications import EfficientNetB7
from keras.models import Sequential, Model
from keras import layers
import tensorflow as tf
from keras.callbacks import EarlyStopping



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


def train_model(model, X, y, batch_size=64, patience=5, validation_split=0.3):
    '''
    Train model with earlystopping and batch size parameters
    Args:

    KwArgs: 

    '''



# def initialise_WagNet():
#     model = Sequential()
