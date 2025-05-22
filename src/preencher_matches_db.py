import pandas as pd
import joblib
import sqlite3
import os

def preencher_matches_db(
    features_path="dados/trusted/features.parquet",
    model_dir="models",
    db_path="matches.db",
    threshold=0.5
):
    """
    Função para preencher a tabela 'matches' do banco SQLite com predições do modelo.

    Parâmetros:
    - features_path: caminho do arquivo parquet com features
    - model_dir: diretório onde ficam os modelos .pkl (pega o mais recente)
    - db_path: caminho do banco SQLite onde a tabela 'matches' será atualizada
    - threshold: valor para classificar como match (default 0.5)
    """
    # Carrega os dados
    df = pd.read_parquet(features_path)

    # Adiciona composite_score (mesma fórmula do treino)
    df['composite_score'] = (
        0.5 * df['skills_similarity'].astype(float) +
        0.3 * df['area_match'].astype(float) +
        0.2 * df['tfidf_similarity'].astype(float)
    )

    # Remove colunas que não fazem parte das features usadas no modelo
    # (essas colunas são usadas só para identificação e labels)
    colunas_para_remover = ["match", "vaga_id", "codigo_profissional"]
    df_features = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])

    # Carrega o modelo mais recente
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pkl")])
    if not model_files:
        raise FileNotFoundError(f"Nenhum modelo .pkl encontrado em {model_dir}")
    model_path = os.path.join(model_dir, model_files[-1])
    model = joblib.load(model_path)

    # Garante que as colunas estejam na mesma ordem esperada pelo modelo
    feature_cols = model.get_booster().feature_names
    X = df_features[feature_cols]

    # Realiza a predição
    probs = model.predict_proba(X)[:, 1]
    matches = (probs >= threshold).astype(int)

    # Extrai ids para inserir no banco
    vaga_ids = df["vaga_id"].astype(str)
    candidato_ids = df["codigo_profissional"].astype(str)

    # Conecta ao banco SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Cria a tabela matches se não existir
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vaga_id TEXT,
            candidato_id TEXT,
            match_result INTEGER,
            probability REAL,
            details TEXT
        )
    """)

    # Insere os registros no banco
    for vaga_id, candidato_id, match, prob in zip(vaga_ids, candidato_ids, matches, probs):
        cursor.execute("""
            INSERT INTO matches (vaga_id, candidato_id, match_result, probability, details)
            VALUES (?, ?, ?, ?, ?)
        """, (vaga_id, candidato_id, int(match), float(prob), None))

    conn.commit()
    conn.close()

    print(f"✅ Banco '{db_path}' populado com {len(df)} registros de matches.")

if __name__ == "__main__":
    preencher_matches_db()
