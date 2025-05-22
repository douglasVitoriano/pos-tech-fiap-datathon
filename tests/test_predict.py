import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi.testclient import TestClient
from api_ml_job_matching import app

client = TestClient(app)

def test_predict():
    payload = {
        "skills_similarity": 0.7,
        "area_match": 1,
        "tfidf_similarity": 0.6,
        "n_cv_skills": 5,
        "n_job_skills": 6,
        "skills_coverage_ratio": 0.42,
        "keyword_overlap": 1,
        "composite_score": 0.68,
        "senioridade_junior": 0,
        "senioridade_pleno": 1,
        "senioridade_senior": 0,
        "senioridade_nao_definido": 0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "match" in response.json()
    assert "probability" in response.json()
