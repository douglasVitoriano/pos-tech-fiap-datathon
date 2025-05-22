
import pandas as pd
import numpy as np
import os
import re
import logging
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from nltk.corpus import stopwords
import nltk
import gc
from tqdm import tqdm

nltk.download('stopwords')
nlp = spacy.load("pt_core_news_sm", disable=['parser', 'ner'])
STOPWORDS = set(stopwords.words('portuguese')).union(set(stopwords.words('english')))
logging.basicConfig(filename='feature_engineering.log', level=logging.INFO)

class FeatureEngineeringPipeline:
    def __init__(self, batch_size=500):
        self.vectorizer = TfidfVectorizer(min_df=5, max_features=2000, ngram_range=(1, 2))
        self.batch_size = batch_size

    def _ensure_consistent_types(self, df, column_name, default_type=str):
        if column_name in df.columns:
            df[column_name] = df[column_name].astype(default_type)
        return df

    def preprocess_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        text = unidecode(text.lower())
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
        return " ".join(tokens)

    def extract_skills(self, text, area=None):
        if not isinstance(text, str) or not text.strip():
            return ""
        skills = set()
        doc = nlp(text)
        skill_verbs = ['saber', 'conhecer', 'dominar', 'experiência', 'habilidade']
        for token in doc:
            if token.text in skill_verbs:
                skills.update([child.text for child in token.children if child.pos_ == 'NOUN'])
        return ", ".join(skills) if skills else ""

    def _process_in_batches(self, data, process_fn, desc=""):
        results = []
        for i in tqdm(range(0, len(data), self.batch_size), desc=desc):
            batch = data.iloc[i:i + self.batch_size]
            results.extend(process_fn(batch))
            gc.collect()
        return results

    def _calculate_similarity_batches(self, texts):
        self.vectorizer.fit(texts[:min(len(texts), 10000)])
        similarities = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Calculando similaridades"):
            batch = texts[i:i + self.batch_size]
            batch_matrix = self.vectorizer.transform(batch)
            sim = cosine_similarity(batch_matrix, batch_matrix).diagonal()
            similarities.extend(sim.tolist())
            del batch_matrix, sim
            gc.collect()
        if len(similarities) != len(texts):
            raise ValueError(f"Incompatibilidade: {len(similarities)} similaridades para {len(texts)} textos.")
        return similarities

    def _normalizar_area(self, area):
        if not isinstance(area, str):
            return None
        area = unidecode(area.strip().lower())
        if "ti" in area:
            return "TI"
        if "financeir" in area or "contabil" in area:
            return "Financeiro"
        if "admin" in area:
            return "Administrativo"
        if "logist" in area:
            return "Logística"
        return area.title()

    def _inferir_area_atuacao_por_cv(self, cv_text):
        if not isinstance(cv_text, str):
            return None
        texto = unidecode(cv_text.lower())
        if "projeto" in texto or "gerenciamento" in texto:
            return "TI - Projetos"
        if "infraestrutura" in texto or "rede" in texto or "servidores" in texto:
            return "TI - Infraestrutura"
        if "desenvolvimento" in texto or "programacao" in texto:
            return "TI - Desenvolvimento/Programação"
        if "suporte" in texto:
            return "TI - Suporte"
        if "financeiro" in texto or "contabil" in texto:
            return "Financeira"
        if "administrativo" in texto:
            return "Administrativa"
        if "comercial" in texto or "vendas" in texto:
            return "Comercial"
        if "recursos humanos" in texto or "rh" in texto:
            return "RH"
        return None

    def generate_features(self, applicants, jobs, prospects):
        try:
            logging.info("Iniciando feature engineering")
            applicants = applicants.copy()
            jobs = jobs.copy()

            applicants['codigo_profissional'] = applicants['infos_basicas'].apply(
                lambda x: x.get('codigo_profissional') if isinstance(x, dict) else None
            )
            applicants['area_atuacao'] = applicants['informacoes_profissionais'].apply(
                lambda x: x.get('area_atuacao') if isinstance(x, dict) else None
            )
            applicants['full_cv'] = applicants.get('cv_pt', pd.Series('')) + " " + applicants.get('cv_en', pd.Series(''))

            applicants['area_atuacao'] = applicants.apply(
                lambda row: row['area_atuacao'] if pd.notnull(row['area_atuacao'])
                else self._inferir_area_atuacao_por_cv(row.get('cv_pt', '')),
                axis=1
            )

            jobs['vaga_id'] = jobs.index.astype(str)
            jobs['area_vaga'] = jobs['perfil_vaga'].apply(
                lambda x: x.get('areas_atuacao') if isinstance(x, dict) else None
            )

            applicants['area_atuacao'] = applicants['area_atuacao'].apply(self._normalizar_area)
            jobs['area_vaga'] = jobs['area_vaga'].apply(self._normalizar_area)

            applicants['processed_cv'] = self._process_in_batches(
                applicants['full_cv'],
                lambda batch: [self.preprocess_text(str(x)) for x in batch],
                "Processando CVs"
            )
            jobs['processed_desc'] = self._process_in_batches(
                jobs['perfil_vaga'].apply(lambda x: x.get('principais_atividades') if isinstance(x, dict) else None),
                lambda batch: [self.preprocess_text(str(x)) for x in batch],
                "Processando vagas"
            )

            applicants['skills'] = self._process_in_batches(
                applicants,
                lambda batch: [self.extract_skills(row['processed_cv'], row['area_atuacao']) for _, row in batch.iterrows()],
                "Extraindo skills dos candidatos"
            )
            jobs['job_skills'] = self._process_in_batches(
                jobs,
                lambda batch: [self.extract_skills(row['processed_desc'], row['area_vaga']) for _, row in batch.iterrows()],
                "Extraindo skills das vagas"
            )

            positives = []
            for vaga_id, row in prospects.iterrows():
                for prospect in row.get('prospects', []):
                    candidate_id = str(prospect.get('codigo', ''))
                    if candidate_id:
                        positives.append({'codigo_profissional': candidate_id, 'vaga_id': str(vaga_id), 'match': 1})

            n_negatives = min(len(positives), 50000)
            negatives = []
            candidate_ids = applicants['codigo_profissional'].dropna().unique()
            job_ids = jobs['vaga_id'].unique()
            existing_pairs = set((p['codigo_profissional'], p['vaga_id']) for p in positives)
            for _ in tqdm(range(n_negatives), desc="Gerando pares negativos"):
                candidate = str(np.random.choice(candidate_ids))
                job = str(np.random.choice(job_ids))
                if (candidate, job) not in existing_pairs:
                    negatives.append({'codigo_profissional': candidate, 'vaga_id': job, 'match': 0})
                    existing_pairs.add((candidate, job))

            pairs = pd.concat([pd.DataFrame(positives), pd.DataFrame(negatives)])
            applicants = self._ensure_consistent_types(applicants, 'codigo_profissional')
            jobs = self._ensure_consistent_types(jobs, 'vaga_id')
            features = pairs.merge(
                applicants[['codigo_profissional', 'processed_cv', 'skills', 'area_atuacao']],
                on='codigo_profissional', how='left'
            ).merge(
                jobs[['vaga_id', 'processed_desc', 'job_skills', 'area_vaga']],
                on='vaga_id', how='left'
            )

            features['combined_text'] = features['processed_cv'].fillna('') + " " + features['processed_desc'].fillna('')
            #features['tfidf_similarity'] = self._calculate_similarity_batches(features['combined_text'].tolist())
            features['tfidf_similarity'] = self._calculate_similarity_pairs(
                features['processed_cv'].tolist(),
                features['processed_desc'].tolist()
            )

            def safe_split(text):
                return set(str(text).split(', ')) if pd.notnull(text) else set()

            features['skills_similarity'] = features.apply(
                lambda x: len(safe_split(x['skills']) & safe_split(x['job_skills'])) / max(len(safe_split(x['job_skills'])), 1),
                axis=1
            )

            def normalizar_areas_multiplas(area):
                if not isinstance(area, str):
                    return set()
                return set(unidecode(area.lower()).split(','))

            features['area_match'] = features.apply(
                lambda x: int(
                    pd.notnull(x['area_atuacao']) and pd.notnull(x['area_vaga']) and
                    len(normalizar_areas_multiplas(x['area_atuacao']) & normalizar_areas_multiplas(x['area_vaga'])) > 0
                ),
                axis=1
            )

            # Novas features
            features['n_cv_skills'] = features['skills'].apply(lambda x: len(safe_split(x)))
            features['n_job_skills'] = features['job_skills'].apply(lambda x: len(safe_split(x)))
            features['skills_coverage_ratio'] = features['skills_similarity'] * features['n_job_skills']

            def infer_seniority(text):
                text = str(text).lower()
                if 'senior' in text:
                    return 'senior'
                elif 'pleno' in text:
                    return 'pleno'
                elif 'junior' in text:
                    return 'junior'
                return 'nao_definido'

            features['senioridade'] = features['processed_cv'].apply(infer_seniority)
            features = pd.get_dummies(features, columns=['senioridade'], prefix='senioridade')

            features['keyword_overlap'] = features.apply(
                lambda x: len(set(str(x['processed_cv']).split()) & set(str(x['processed_desc']).split())),
                axis=1
            )

            cols_to_keep = [
                'codigo_profissional', 'vaga_id',
                'tfidf_similarity', 'skills_similarity', 'area_match',
                'n_cv_skills', 'n_job_skills', 'skills_coverage_ratio',
                'keyword_overlap', 'match'
            ] + [col for col in features.columns if col.startswith('senioridade_')]

            final_features = features[cols_to_keep].dropna()

            os.makedirs("dados/trusted", exist_ok=True)
            final_features.to_parquet("dados/trusted/features.parquet", index=False)
            logging.info("Features geradas com sucesso.")
            return final_features, "Features geradas com sucesso!"
        except Exception as e:
            logging.exception("Erro ao gerar features")
            return pd.DataFrame(), f"Erro: {str(e)}"


if __name__ == "__main__":
    applicants = pd.read_json("dados/raw/applicants.json", orient="index")
    jobs = pd.read_json("dados/raw/vagas.json", orient="index")
    prospects = pd.read_json("dados/raw/prospects.json", orient="index")

    pipeline = FeatureEngineeringPipeline(batch_size=500)
    features, msg = pipeline.generate_features(applicants, jobs, prospects)
    print(msg)
    if not features.empty:
        print("\nAmostra de features:")
        print(features.sample(5))
