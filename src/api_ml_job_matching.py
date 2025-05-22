from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import glob

app = FastAPI()

# Carregar o modelo mais recente
try:
    latest_model = sorted(glob.glob("models/model_*.pkl"))[-1]
    model = joblib.load(latest_model)
    model_features = model.get_booster().feature_names  # pega ordem correta
except Exception as e:
    raise RuntimeError("❌ Erro ao carregar modelo") from e

# Modelo de input
class MatchInput(BaseModel):
    skills_similarity: float
    area_match: int
    tfidf_similarity: float
    n_cv_skills: int
    n_job_skills: int
    skills_coverage_ratio: float
    keyword_overlap: int
    composite_score: float
    senioridade_junior: int
    senioridade_pleno: int
    senioridade_senior: int
    senioridade_nao_definido: int

@app.post("/predict")
def predict_match(data: MatchInput):
    df = pd.DataFrame([data.model_dump()])

    # Ordenar colunas exatamente como esperado pelo modelo
    try:
        df = df[model_features]
    except Exception as e:
        raise RuntimeError(f"Erro ao reordenar colunas do input: {e}")

    try:
        proba = model.predict_proba(df)[0][1]
        match = int(proba >= 0.5)
        return {"match": match, "probability": float(proba)}
    except Exception as e:
        raise RuntimeError(f"Erro ao fazer predição: {e}")
