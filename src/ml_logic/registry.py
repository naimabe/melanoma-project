# from src.ml_logic.params import LOCAL_REGISTRY_PATH

import glob
import os
import time
import pickle
import tensorflow

from tensorflow.keras import Model, models


def save_model(model: Model = None) -> None:
    """
    persist trained model, params and metrics
    """
    # save model
    if model is not None:
        model_path = os.path.join(os.environ.get('LOCAL_REGISTRY_PATH'), "models")
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None


def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """

    model_path = os.path.join(os.environ.get('LOCAL_REGISTRY_PATH'), "models")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
