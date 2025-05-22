from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import glob
import sqlite3
from datetime import datetime
from typing import Optional, List

app = FastAPI(
    title="Job Matching API",
    description="API para prever e consultar matches entre vagas e candidatos",
    version="2.1.0"
)

# --- Configuração do Banco de Dados ---
def init_db():
    conn = sqlite3.connect("matches.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vaga_id TEXT NOT NULL,
            candidato_id TEXT NOT NULL,
            match_result INTEGER NOT NULL,
            probability REAL NOT NULL,
            details TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Índice para consultas por vaga_id
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vaga_id ON matches(vaga_id)")
    conn.commit()
    conn.close()

init_db()

# --- Modelo de Machine Learning ---
def load_latest_model():
    try:
        model_files = glob.glob("models/model_*.pkl")
        if not model_files:
            raise FileNotFoundError("Nenhum modelo encontrado na pasta 'models'")
        latest_model = sorted(model_files)[-1]
        model = joblib.load(latest_model)
        model_features = model.get_booster().feature_names
        return model, model_features
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")

try:
    model, model_features = load_latest_model()
except Exception as e:
    raise RuntimeError(f"Falha ao iniciar API: {str(e)}")

# --- Schemas ---
class MatchInput(BaseModel):
    vaga_id: str
    candidato_id: str
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
    metadata: Optional[dict] = None

class MatchResult(BaseModel):
    vaga_id: str
    candidato_id: str
    match: int
    probability: float
    timestamp: str

class CandidatoMatch(BaseModel):
    candidato_id: str
    probability: float
    timestamp: str
    metadata: Optional[dict] = None

class VagaMatchInfo(BaseModel):
    vaga_id: str
    total_matches: int
    best_candidate_prob: float
    last_match_time: str
    candidates: Optional[List[str]] = None  # Opcional: IDs dos candidatos

# --- Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=MatchResult)
async def predict_match(data: MatchInput):
    """Registra um novo candidato-vaga e retorna o resultado do match"""
    try:
        input_data = data.model_dump()
        metadata = input_data.pop('metadata', None)
        
        df = pd.DataFrame([input_data])
        df = df[model_features]
        
        proba = model.predict_proba(df)[0][1]
        match_result = int(proba >= 0.5)
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect("matches.db")
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO matches 
            (vaga_id, candidato_id, match_result, probability, details) 
            VALUES (?, ?, ?, ?, ?)""",
            (data.vaga_id, data.candidato_id, match_result, float(proba), 
             str(metadata) if metadata else None)
        )
        conn.commit()
        conn.close()

        return {
            "vaga_id": data.vaga_id,
            "candidato_id": data.candidato_id,
            "match": match_result,
            "probability": float(proba),
            "timestamp": timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/matches/candidato/{candidato_id}", response_model=List[MatchResult])
async def get_matches_by_candidato_id(candidato_id: str):
    """Consulta todos os matches de um candidato específico"""
    try:
        conn = sqlite3.connect("matches.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT vaga_id, candidato_id, match_result, probability, timestamp
            FROM matches
            WHERE candidato_id = ? AND match_result = 1
            ORDER BY probability DESC
        """, (candidato_id,))

        results = [
            {
                "vaga_id": row[0],
                "candidato_id": row[1],
                "match": row[2],
                "probability": row[3],
                "timestamp": row[4]
            } for row in cursor.fetchall()
        ]
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")

@app.get("/matches/vaga/{vaga_id}", response_model=List[CandidatoMatch])
async def get_matches_by_vaga_id(vaga_id: str):
    """Consulta todos os matches de uma vaga específica"""
    try:
        conn = sqlite3.connect("matches.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT candidato_id, probability, timestamp, details
            FROM matches
            WHERE vaga_id = ? AND match_result = 1
            ORDER BY probability DESC
        """, (vaga_id,))

        results = [
            {
                "candidato_id": row[0],
                "probability": row[1],
                "timestamp": row[2],
                "metadata": eval(row[3]) if row[3] else None
            } for row in cursor.fetchall()
        ]
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")

@app.get("/debug/vagas_salvas")
async def debug_vagas_salvas():
    conn = sqlite3.connect("matches.db")
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT vaga_id FROM matches ORDER BY vaga_id")
    results = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return {"vaga_ids_salvos": results}

@app.get("/vagas/matches", response_model=List[VagaMatchInfo])
async def get_vagas_com_matches(
    min_prob: float = Query(0.5, description="Probabilidade mínima para considerar match"),
    min_matches: int = Query(1, description="Número mínimo de matches para incluir"),
    include_candidates: bool = Query(False, description="Incluir IDs dos candidatos")
):
    """Consulta avançada de vagas com matches"""
    conn = None
    try:
        conn = sqlite3.connect("matches.db")
        cursor = conn.cursor()
        
        # Query básica de agregação
        query = """
            SELECT 
                vaga_id,
                COUNT(*) as total_matches,
                MAX(probability) as best_prob,
                MAX(timestamp) as last_time
            FROM matches
            WHERE match_result = 1 AND probability >= ?
            GROUP BY vaga_id
            HAVING COUNT(*) >= ?
            ORDER BY total_matches DESC
        """
        cursor.execute(query, (min_prob, min_matches))
        
        vagas = []
        for row in cursor.fetchall():
            vaga_info = {
                "vaga_id": row[0],
                "total_matches": row[1],
                "best_candidate_prob": row[2],
                "last_match_time": row[3]
            }
            
            if include_candidates:
                cursor.execute(
                    "SELECT candidato_id FROM matches WHERE vaga_id = ? AND match_result = 1 AND probability >= ?",
                    (row[0], min_prob)
                )
                vaga_info["candidates"] = [c[0] for c in cursor.fetchall()]
            
            vagas.append(vaga_info)
        
        return vagas

    except sqlite3.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no banco de dados: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

@app.get("/debug/check_data")
async def debug_check_data():
    conn = sqlite3.connect("matches.db")
    cursor = conn.cursor()
    
    # Verifica tabelas existentes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    # Verifica registros na tabela matches
    cursor.execute("SELECT COUNT(*) FROM matches")
    count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "tables": [t[0] for t in tables],
        "total_matches": count,
        "model_loaded": bool(model)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": bool(model)}