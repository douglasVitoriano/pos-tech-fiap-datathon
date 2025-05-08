from data_preparation import DataPreparation
from DataIngestorGlue import DataIngestor
import pandas as pd

import sys

ingestor = DataIngestor(bucket_name='fiap-datathon-mlet2/trusted')
merger = DataPreparation()

df_vagas = ingestor.read_parquet('vagas.parquet')
df_prospects = ingestor.read_parquet('prospects.parquet')
df_applicants = ingestor.read_parquet('applicants.parquet')

# Merge entre vagas e prospects
df_vagas_prospects = merger.merge(df_vagas,df_prospects, left_on= 'vaga_id',right_on='codigo_vaga', how='inner')

# Merge entre vagas_prospects e applicants
df_vagas_prospects_applicants = merger.merge(df_vagas_prospects, df_applicants, left_on='codigo_applicants', 
                                             right_on='codigo_candidato', how='inner')

# tipagem
#datetime
df_vagas_prospects_applicants['data_requicisao'] = pd.to_datetime(df_vagas_prospects_applicants['data_requicisao'], errors='coerce')
df_vagas_prospects_applicants['limite_esperado_para_contratacao'] = pd.to_datetime(df_vagas_prospects_applicants['limite_esperado_para_contratacao'], errors='coerce')
df_vagas_prospects_applicants['prazo_contratacao'] = pd.to_datetime(df_vagas_prospects_applicants['prazo_contratacao'], errors='coerce')
df_vagas_prospects_applicants['data_inicial'] = pd.to_datetime(df_vagas_prospects_applicants['data_inicial'], errors='coerce')
df_vagas_prospects_applicants['data_final'] = pd.to_datetime(df_vagas_prospects_applicants['data_final'], errors='coerce')
df_vagas_prospects_applicants['data_candidatura'] = pd.to_datetime(df_vagas_prospects_applicants['data_candidatura'], errors='coerce')
df_vagas_prospects_applicants['ultima_atualizacao'] = pd.to_datetime(df_vagas_prospects_applicants['ultima_atualizacao'], errors='coerce')

#str
df_vagas_prospects_applicants['codigo_applicants'] = df_vagas_prospects_applicants['codigo_applicants'].astype(str)
df_vagas_prospects_applicants['codigo_candidato'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)
df_vagas_prospects_applicants['vaga_id'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)
df_vagas_prospects_applicants['codigo_vaga'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)
df_vagas_prospects_applicants['titulo'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)
df_vagas_prospects_applicants['modalidade'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)
df_vagas_prospects_applicants['nome_applicants'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)
df_vagas_prospects_applicants['situacao_candidado'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)
df_vagas_prospects_applicants['recrutador'] = df_vagas_prospects_applicants['codigo_candidato'].astype(str)

#numeric
df_vagas_prospects_applicants['valor_venda'] = pd.to_numeric(df_vagas_prospects_applicants['valor_venda'], errors='coerce')
df_vagas_prospects_applicants['valor_compra_1'] = pd.to_numeric(df_vagas_prospects_applicants['valor_venda'], errors='coerce')
df_vagas_prospects_applicants['valor_compra_2'] = pd.to_numeric(df_vagas_prospects_applicants['valor_venda'], errors='coerce')

#feature engineering
df_vagas_prospects_applicants['data_final_nulo'] = df_vagas_prospects_applicants['data_final'].isnull()
correlacao_data_final = df_vagas_prospects_applicants.groupby('situacao_candidado')['data_final_nulo'].mean()

#correlacao data_final
correlacao_data_final_df = correlacao_data_final.reset_index()
correlacao_data_final_df.columns = ['situacao_candidado', 'proporcao_data_final_nulo']

#Duração_da_vaga
df_vagas_prospects_applicants['duracao_vaga'] = (
    df_vagas_prospects_applicants['data_final'] - df_vagas_prospects_applicants['data_inicial']
).dt.days
#Duração da candidatura
df_vagas_prospects_applicants['vaga_aberta'] = df_vagas_prospects_applicants['data_final'].isnull()

#correlacao data_inicial
df_vagas_prospects_applicants['data_inicial_nulo'] = df_vagas_prospects_applicants['data_inicial'].isnull()
correlacao_data_incial = df_vagas_prospects_applicants.groupby('situacao_candidado')['data_inicial_nulo'].mean()





