import sqlite3
import random
from datetime import datetime, timedelta

def init_test_data():
    conn = sqlite3.connect("matches.db")
    cursor = conn.cursor()
    
    # Limpa tabela existente
    cursor.execute("DELETE FROM matches")
    
    # Insere 50 matches de teste
    for i in range(1, 51):
        vaga_id = f"VAGA-{i%10+1}"  # 10 vagas diferentes
        candidato_id = f"CAND-{i}"
        probability = random.uniform(0.4, 0.95)
        match_result = 1 if probability > 0.5 else 0
        timestamp = (datetime.now() - timedelta(days=random.randint(0,30))).isoformat()
        
        cursor.execute(
            "INSERT INTO matches (vaga_id, candidato_id, match_result, probability, timestamp) VALUES (?, ?, ?, ?, ?)",
            (vaga_id, candidato_id, match_result, probability, timestamp)
        )
    
    conn.commit()
    conn.close()
    print("✅ Dados de teste inseridos com sucesso")

if __name__ == "__main__":
    init_test_data()