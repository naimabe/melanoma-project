import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential


def load_Model_G():
    '''
    Cette fonction instantie un modèle EfficientNet pre-entrainé et y ajoute des layers Dense
    Elle sera utilisée pour prédire sur la data sous forme d'images.
    '''

    def load_efficientnet():
        '''
        load pre-trained model
        Args: None
        return: pre-trained model
        '''
        model = EfficientNetV2B0(include_top=False,
                            weights='imagenet',
                            input_shape=(64, 64, 3))
        return model

    def set_nontrainable_layers(model):
        '''
        Ensures pre-trained model is not trainable on new data.
        Args: None
        return: pre-trained model
        '''
        model.trainable = False
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
        prediction_layer = layers.Dense(9, activation='softmax')

        model = Sequential([base_model,
                            flattening_layer,
                            prediction_layer])

        model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(learning_rate=float(os.environ.get('LEARNING_RATE'))),
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
    model = models.Sequential()
    model.add(layers.Dense(9, activation='softmax', input_dim=11))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
