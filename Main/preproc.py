
import shutil
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np



def move_images():
    df = pd.read_csv(Path('..', 'data', 'Skin_lesion', 'ISIC_2019_Training_GroundTruth.csv'))
    df.set_index('image')
    for source in df.index:
        for column in df.columns:
            if df.loc[source][column] == 1:
                source_path = Path('..', 'data', 'Skin_lesion', 'ISIC_2019_Training_Input', f'{source}.jpg')
                destination_path = Path('..', 'data', f'{column}')
                shutil.move(source_path, destination_path)






