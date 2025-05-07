import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelBinarizer

def normalize_text(text):
    if isinstance(text, str):
        return text.lower().strip()
    return ""

def compute_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit([text1, text2])
    vec1 = vectorizer.transform([text1])
    vec2 = vectorizer.transform([text2])
    sim = cosine_similarity(vec1, vec2)
    return sim[0][0]

def extract_match_pairs(df_prospects):
    matches = []
    for _, row in df_prospects.iterrows():
        vaga_id = row['vaga_id']
        for prospect in row['prospects']:
            candidato_id = prospect['codigo']
            matches.append({
                'codigo_profissional': candidato_id,
                'vaga_id': vaga_id,
                'match': 1
            })
    return pd.DataFrame(matches)

def create_features(df_applicants, df_vagas, df_prospects):
    # Pré-processa applicants
    applicants = pd.json_normalize(df_applicants)
    applicants['codigo_profissional'] = applicants['infos_basicas.codigo_profissional']
    applicants['skills'] = applicants['informacoes_profissionais.conhecimentos_tecnicos'].apply(normalize_text)
    applicants['area_atuacao'] = applicants['informacoes_profissionais.area_atuacao'].apply(normalize_text)
    applicants['nivel_ingles'] = applicants['formacao_e_idiomas.nivel_ingles'].apply(normalize_text)
    applicants['nivel_academico'] = applicants['formacao_e_idiomas.nivel_academico'].apply(normalize_text)

    # Pré-processa vagas
    vagas = pd.json_normalize(df_vagas)
    vagas['vaga_id'] = vagas.index.astype(str)
    vagas['competencias'] = vagas['perfil_vaga.competencia_tecnicas_e_comportamentais'].apply(normalize_text)
    vagas['areas_atuacao'] = vagas['perfil_vaga.areas_atuacao'].apply(normalize_text)
    vagas['nivel_ingles'] = vagas['perfil_vaga.nivel_ingles'].apply(normalize_text)
    vagas['nivel_academico'] = vagas['perfil_vaga.nivel_academico'].apply(normalize_text)

    # Produto cartesiano candidatos x vagas
    applicants['key'] = 1
    vagas['key'] = 1
    df_cartesian = applicants.merge(vagas, on='key').drop('key', axis=1)

    # Features
    df_cartesian['similaridade_skills'] = df_cartesian.apply(
        lambda row: compute_cosine_similarity(row['skills'], row['competencias']), axis=1)

    df_cartesian['match_ingles'] = (df_cartesian['nivel_ingles_x'] == df_cartesian['nivel_ingles_y']).astype(int)
    df_cartesian['match_academico'] = (df_cartesian['nivel_academico_x'] == df_cartesian['nivel_academico_y']).astype(int)

    # Variável alvo (match)
    df_prospects['vaga_id'] = df_prospects.index.astype(str)
    matches_df = extract_match_pairs(df_prospects)

    df_final = df_cartesian.merge(matches_df, how='left', on=['codigo_profissional', 'vaga_id'])
    df_final['match'] = df_final['match'].fillna(0).astype(int)

    # Seleciona features finais
    features = df_final[[
        'codigo_profissional', 'vaga_id', 'similaridade_skills', 'match_ingles', 'match_academico', 'match'
    ]]

    return features

if __name__ == "__main__":
    applicants = pd.read_json("dados/raw/applicants.json")
    vagas = pd.read_json("dados/raw/vagas.json")
    prospects = pd.read_json("dados/raw/prospects.json")

    features = create_features(applicants, vagas, prospects)
    features.to_parquet("dados/trusted/features.parquet", index=False)
    print("Features salvas em dados/trusted/features.parquet")