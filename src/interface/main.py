from src.ml_logic.preproc import images_to_dataset, preprocessing_X_tabulaire, move_images_tertiaire
from src.ml_logic.registry import load_model, save_model
from src.ml_logic.model import model_concat, load_Model_G, train_model


if __name__ == '__main__':
    preprocessing_X_tabulaire()
    move_images_tertiaire()
    images_to_dataset()
    train_model(load_model())
