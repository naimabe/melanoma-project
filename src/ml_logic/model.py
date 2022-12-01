import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB7
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential


def initialise_EfficientNet_model(learning_rate):
    '''
    Initialises non-trainable basemodel EfficientNet, adds two layers and compiles for multicategorical classification.
    Args: learning rate (int)
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
                               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
