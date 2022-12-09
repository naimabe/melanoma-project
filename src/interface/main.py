from src.ml_logic.registry import load_model
from src.ml_logic.model import predict_simple
import os

model_path = os.environ.get('BEST_MODEL_PATH')

image_name = 'ISIC_0064021.jpg'

if __name__ == '__main__':
    model = load_model(model_path)
    prediction = predict_simple(model, 'TEST_PATH', image_name)
    print(prediction)
