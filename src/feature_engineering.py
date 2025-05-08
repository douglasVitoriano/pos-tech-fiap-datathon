import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text.lower().strip()

def compute_cosine_similarity(text1, text2):
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    if not text1 and not text2:
        return 1.0  # Ambos vazios = completamente iguais
    if not text1 or not text2:
        return 0.0  # Um vazio = nenhuma similaridade
        
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit([text1, text2])
        vec1 = vectorizer.transform([text1])
        vec2 = vectorizer.transform([text2])
        sim = cosine_similarity(vec1, vec2)
        return float(sim[0][0])
    except ValueError:
        return 0.0

def extract_match_pairs(df_prospects):
    matches = []
    for vaga_id, row in df_prospects.iterrows():
        for prospect in row['prospects']:
            candidato_id = prospect['codigo']
            matches.append({
                'codigo_profissional': candidato_id,
                'vaga_id': str(vaga_id),
                'match': 1
            })
    return pd.DataFrame(matches)

def create_features(df_applicants, df_vagas, df_prospects):
    # Pré-processa applicants
    applicants = pd.json_normalize(df_applicants.to_dict().values())
    applicants['codigo_profissional'] = applicants['infos_basicas.codigo_profissional']
    applicants['skills'] = applicants['informacoes_profissionais.conhecimentos_tecnicos'].apply(normalize_text)
    applicants['area_atuacao'] = applicants['informacoes_profissionais.area_atuacao'].apply(normalize_text)
    applicants['nivel_ingles'] = applicants['formacao_e_idiomas.nivel_ingles'].apply(normalize_text)
    applicants['nivel_academico'] = applicants['formacao_e_idiomas.nivel_academico'].apply(normalize_text)

    # Pré-processa vagas
    vagas = pd.json_normalize(df_vagas.to_dict().values())
    vagas['vaga_id'] = vagas.index.astype(str)
    vagas['competencias'] = vagas['perfil_vaga.competencia_tecnicas_e_comportamentais'].apply(normalize_text)
    vagas['areas_atuacao'] = vagas['perfil_vaga.areas_atuacao'].apply(normalize_text)
    vagas['nivel_ingles'] = vagas['perfil_vaga.nivel_ingles'].apply(normalize_text)
    vagas['nivel_academico'] = vagas['perfil_vaga.nivel_academico'].apply(normalize_text)

    # Extrai matches (positivos)
    df_prospects['vaga_id'] = df_prospects.index.astype(str)
    matches_df = extract_match_pairs(df_prospects)

    df_matches = matches_df.merge(applicants, on='codigo_profissional', how='left')
    df_matches = df_matches.merge(vagas, on='vaga_id', how='left')
    df_matches['match'] = 1

    # Renomeia colunas para evitar conflitos
    df_matches = df_matches.add_suffix('_match')
    df_matches.rename(columns={
        'codigo_profissional_match': 'codigo_profissional',
        'vaga_id_match': 'vaga_id',
        'match_match': 'match'
    }, inplace=True)

    # Amostra de negativos (igual n_positivos)
    n_samples = len(df_matches)
    n_samples = min(n_samples, len(applicants), len(vagas))

    sampled_applicants = applicants.sample(n=n_samples, random_state=42).reset_index(drop=True)
    sampled_vagas = vagas.sample(n=n_samples, random_state=42).reset_index(drop=True)

    sampled_applicants = sampled_applicants.add_suffix('_app')
    sampled_vagas = sampled_vagas.add_suffix('_vaga')

    df_negatives = pd.concat([sampled_applicants, sampled_vagas], axis=1)
    df_negatives['match'] = 0

    # Junta positivos + negativos
    df_final = pd.concat([df_matches, df_negatives], ignore_index=True)

    # === Cria novas features ===

    # Similaridade de skills
    def get_skills(row):
        return row.get('skills_match', row.get('skills_app', ''))
    def get_competencias(row):
        return row.get('competencias_match', row.get('competencias_vaga', ''))
    df_final['similaridade_skills'] = df_final.apply(lambda row:
        compute_cosine_similarity(get_skills(row), get_competencias(row)), axis=1)

    # Similaridade de área de atuação (nova feature)
    def get_area(row):
        return row.get('area_atuacao_match', row.get('area_atuacao_app', ''))
    def get_area_vaga(row):
        return row.get('areas_atuacao_match', row.get('areas_atuacao_vaga', ''))
    df_final['similaridade_area_atuacao'] = df_final.apply(lambda row:
        compute_cosine_similarity(get_area(row), get_area_vaga(row)), axis=1)

    # Similaridade soft para nível de inglês e acadêmico (nova abordagem)
    def soft_match(text1, text2):
        text1 = normalize_text(text1)
        text2 = normalize_text(text2)
        if not text1 and not text2:
            return 1.0  # Ambos vazios = iguais
        if not text1 or not text2:
            return 0.0  # Um vazio = diferente
        if text1 == text2:
            return 1.0
        if text1 in text2 or text2 in text1:
            return 0.5
        return 0.0

    def get_ingles(row):
        return row.get('nivel_ingles_match', row.get('nivel_ingles_app', ''))
    def get_ingles_vaga(row):
        return row.get('nivel_ingles_match', row.get('nivel_ingles_vaga', ''))
    df_final['match_ingles_soft'] = df_final.apply(lambda row:
        soft_match(get_ingles(row), get_ingles_vaga(row)), axis=1)

    def get_academico(row):
        return row.get('nivel_academico_match', row.get('nivel_academico_app', ''))
    def get_academico_vaga(row):
        return row.get('nivel_academico_match', row.get('nivel_academico_vaga', ''))
    df_final['match_academico_soft'] = df_final.apply(lambda row:
        soft_match(get_academico(row), get_academico_vaga(row)), axis=1)

    # Junta features finais
    features = df_final[[  
        'codigo_profissional', 'vaga_id',
        'similaridade_skills', 'similaridade_area_atuacao',
        'match_ingles_soft', 'match_academico_soft',
        'match'
    ]]

    return features

def ensure_directory_exists(path):
    """Garante que o diretório existe, criando se necessário"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    try:
        # Carrega os dados
        applicants = pd.read_json("dados/raw/applicants.json")
        vagas = pd.read_json("dados/raw/vagas.json")
        prospects = pd.json_normalize(pd.read_json("dados/raw/prospects.json").to_dict().values())

        # Processa as features
        features = create_features(applicants, vagas, prospects)
        
        # Define os caminhos de saída
        parquet_path = "dados/trusted/features.parquet"
        csv_path = "dados/trusted/features.csv"
        
        # Garante que o diretório existe
        ensure_directory_exists(parquet_path)
        
        # Tenta salvar em diferentes formatos
        try:
            features.to_parquet(parquet_path, index=False, engine='pyarrow')
            print(f"Features salvas em {parquet_path}")
        except Exception as e:
            print(f"Erro ao salvar como Parquet: {e}")
            try:
                features.to_parquet(parquet_path, index=False, engine='fastparquet')
                print(f"Features salvas em {parquet_path} usando fastparquet")
            except Exception as e:
                print(f"Erro ao salvar como Parquet com fastparquet: {e}")
                features.to_csv(csv_path, index=False)
                print(f"Features salvas como CSV em {csv_path}")
                
    except Exception as e:
        print(f"Erro ao processar features: {e}")
        if 'features' in locals():
            ensure_directory_exists(csv_path)
            features.to_csv(csv_path, index=False)
            print(f"Features parciais salvas como CSV em {csv_path}")
