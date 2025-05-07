from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/model.pkl')

class InputData(BaseModel):
    similaridade_skills: float
    match_ingles: int
    match_academico: int

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    X = pd.DataFrame([input_dict])
    prediction = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "prediction": int(prediction[0]),
        "probability": proba[0].tolist()
    }

@app.get("/")
def root():
    return {"message": "API para predição de match candidato-vaga está no ar."}