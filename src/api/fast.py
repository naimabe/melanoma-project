from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.ml_logic.registry import load_model

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

# def predict(image, str, int, str ) -> str:
#     df,extra_info = request_gopluslab(token_address)
#     model = app.state.model
#     df = clean_nouveau_token(df)
#     status = prediction(model, df, extra_info)
#     report = generate_report(extra_info, status)
#     return report
