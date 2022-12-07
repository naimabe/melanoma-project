from src.ml_logic.preproc import  preprocessing_pipeline
from src.ml_logic.registry import load_model, save_model
from src.ml_logic.model import model_concat, load_Model_G, train_model


if __name__ == '__main__':
    img, data, target  = preprocessing_pipeline('IMAGE_DATA_PATH', jumpfile=1)
    train_model(load_model(), )
