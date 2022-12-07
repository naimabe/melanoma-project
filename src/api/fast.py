from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.ml_logic.registry import load_model
from src.ml_logic.preproc import images_to_dataset

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.state.model = load_model()

@app.get("/")
def root():
    return {"message": "Melanoma prediction assistance"}


@app.get("/predict")
def predict(image):
    model = load_model(path='BEST_MODEL_PATH')
    images_to_dataset(image, validation_split=False)
    
