# 🤖 Job Matching com Machine Learning e FastAPI

Este projeto implementa uma solução completa de **Job Matching** entre candidatos e vagas, utilizando técnicas de Machine Learning, engenharia de atributos, FastAPI e rastreabilidade com MLflow. O objetivo é prever a compatibilidade entre perfis profissionais e oportunidades de forma automatizada, interpretável e reutilizável.

---

## 📂 Estrutura do Projeto

├── feature_engineering.py # Geração das features a partir de currículos e vagas
├── train_model.py # Treinamento do modelo com rastreamento MLflow
├── api_ml_jobmatching.py # API REST com FastAPI para servir o modelo
├── models/ # Modelos salvos (.pkl)
├── metrics/ # Métricas salvas em JSON
├── visuals/ # Gráficos de avaliação
├── artifacts/ # Análises de erro (falsos positivos/negativos)
├── matches.db # Banco SQLite com registros de matches


---

## ⚙️ Pipeline de Machine Learning

### 1. 🔬 Engenharia de Atributos (`feature_engineering.py`)
- **Limpeza e normalização** de currículos e descrições de vagas
- **Extração de skills** mencionadas nos textos
- **Inferência de área de atuação** (ex: TI, RH)
- **Cálculo de similaridades**:
  - TF-IDF entre currículo e vaga
  - Skills (interseção)
  - Áreas profissionais
- **Geração de features** como:
  - Número de skills no CV e na vaga
  - Proporção de cobertura
  - Sobreposição de palavras-chave
  - Inferência de senioridade (junior, pleno, sênior)
- **Saída**: `features.parquet` com colunas como:
  - `tfidf_similarity`, `skills_similarity`, `area_match`
  - `keyword_overlap`, `composite_score`
  - `senioridade_*`, `match` (rótulo alvo)

---

### 2. 🧠 Treinamento do Modelo (`train_model.py`)
- **Dados**: Lê `features.parquet`
- **Pré-processamento** e balanceamento com **SMOTE**
- **Modelo**: `XGBoost` com `RandomizedSearchCV`
- **Métricas**:
  - AUC ROC, F1 Score, matriz de confusão, classification report
- **Análise de erros**: Falsos positivos/negativos
- **Salvamento**:
  - Modelo `.pkl` via `joblib`
  - Registro completo via **MLflow**
  - Gráficos de importância de features

---

## 🌐 API REST com FastAPI (`api_ml_jobmatching.py`)

### 🔍 Carregamento do Modelo
Ao iniciar, a API carrega automaticamente o modelo `.pkl` mais recente salvo em `models/`.

### 📥 Schema de Entrada (`MatchInput`)
```json
{
  "skills_similarity": 0.78,
  "area_match": 1,
  "tfidf_similarity": 0.65,
  "n_cv_skills": 5,
  "n_job_skills": 8,
  "skills_coverage_ratio": 0.62,
  "keyword_overlap": 4,
  "composite_score": 0.73,
  "senioridade_junior": 0,
  "senioridade_pleno": 1,
  "senioridade_senior": 0,
  "senioridade_nao_definido": 0
}

## 📤 Retorno do endpoint /predict

{
  "match": 1,
  "probability": 0.82
}

🌐 Endpoints disponíveis
GET / → Redireciona para /docs (Swagger)

POST /predict → Retorna probabilidade de match entre candidato e vaga

GET /matches/candidato/{candidato_id} → Retorna todos os matches para um candidato

GET /matches/vaga/{vaga_id} → Retorna todos os matches para uma vaga

GET /vagas/matches → Lista vagas com matches, com filtros como:

min_prob

min_matches

include_candidates=true|false

💾 Banco de Dados (SQLite)
A API utiliza o arquivo matches.db para armazenar e consultar os resultados de previsões de match realizadas anteriormente.

🧪 Execução Local
1. Feature Engineering
bash
Copiar
Editar
python feature_engineering.py
2. Treinamento do modelo
bash
Copiar
Editar
python train_model.py
3. Subir a API
bash
Copiar
Editar
uvicorn api_ml_jobmatching:app --reload
Acesse a documentação interativa:

bash
Copiar
Editar
http://localhost:8000/docs
📈 Treinamento com MLflow
Exemplo de uso básico do MLflow:

python
Copiar
Editar
import mlflow
mlflow.set_experiment("teste_mlflow")
with mlflow.start_run():
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "random_forest_model")

### 🧰 Requisitos
Python 3.8+

FastAPI

Pandas

Scikit-learn

XGBoost

Joblib

MLflow

Uvicorn

Imbalanced-learn (SMOTE)

SQLite3

Instalação:

bash
Copiar
Editar
pip install -r requirements.txt
👨‍💻 Autor
