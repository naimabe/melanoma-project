from keras.applications import EfficientNetB7
from keras.models import Sequential, Model
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
    base_model.add(layers.Dense(9, activation='softmax'))
    model = base_model.compile(loss='categorical_crossentropy',
                               optimizer=tf.keras.optimizers.Adam,
                               metrics=['accuracy', 'precision'])
    return model



# def initialise_WagNet():
#     model = Sequential()
