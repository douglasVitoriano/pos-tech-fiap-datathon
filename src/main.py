from DataIngestor import DataIngestor
from DataProcessor import DataProcessor
import pandas as pd

def load_data():
    try:
        #========================== Ingestão===============================================
        ingestor = DataIngestor()

        df_applicants_raw = ingestor.read_json("dados/raw/applicants.json").T
        df_prospects_raw = ingestor.read_json("dados/raw/prospects.json").T
        df_vagas_raw = ingestor.read_json("dados/raw/vagas.json").T

        #========================= Validação Schema==========================================

        # Transforma as colunas do df para dict para uso na metodo validate_schema
        applicants_schema_raw = {col: str(dtype) for col, dtype in df_applicants_raw.dtypes.items()}
        prospects_schema_raw = {col: str(dtype) for col, dtype in df_prospects_raw.dtypes.items()}
        vagas_schema_raw = {col: str(dtype) for col, dtype in df_vagas_raw.dtypes.items()}

        # ingestor.df = df_applicants_raw
        # ingestor.df = prospects_schema_raw
        # ingestor.df = vagas_schema_raw
        ingestor.validate_schema(df_applicants_raw, applicants_schema_raw)
        ingestor.validate_schema(df_prospects_raw, prospects_schema_raw)
        ingestor.validate_schema(df_vagas_raw, vagas_schema_raw)

        return df_applicants_raw, df_prospects_raw, df_vagas_raw, ingestor
     
    except ValueError as e:
        print(f"Erro de validação: {e}")
        return None, None, None
    except TypeError as e:
        print(f"Erro de tipo de dado: {e}")
        return None, None, None

def processing_data(df_applicants_raw, df_prospects_raw, df_vagas_raw,ingestor):

    try:
        #========================= Processamento=====================================
        
       #Vagas
        vagas_processor = DataProcessor(df_vagas_raw)
        df_vagas_processed = (
            vagas_processor
            .reset_index()
            .rename_column({"index": "vaga_id"})
            .json_normalize(['informacoes_basicas', 'perfil_vaga', 'beneficios'])
            .drop_columns(['informacoes_basicas', 'perfil_vaga', 'beneficios'])
            .get_df()
        )
        ingestor.save_to_parquet(df_vagas_processed, "dados/trusted/vagas.parquet")
    except Exception as e:
        print(f"Erro durante o processamento dos dados de vagas: {e}")

    #Prospects
    try:
        prospects_processor = DataProcessor(df_prospects_raw)
        df_prospects_processed = (
            prospects_processor
            .explode_column("prospects")
            .reset_index()
            .json_normalize(["prospects"])
            .rename_column({"index": "codigo_vaga","codigo": "codigo_applicants", "nome": "nome_applicants"})
            .drop_columns(["prospects"])
            .get_df()

        )
        print(df_prospects_processed)

        ingestor.save_to_parquet(df_prospects_processed, "dados/trusted/prospects.parquet")
    except Exception as e:
        print(f"Erro durante o processamento dos dados de prospects: {e}")

    #applicants
    try:
        applicants_processor = DataProcessor(df_applicants_raw)
        df_applicants_processed = (
            applicants_processor
            .reset_index()
            .rename_column({"index": "codigo_candidato", "cargo_atual": "cargo_atual_old"})
            .json_normalize(["infos_basicas", "informacoes_pessoais", "informacoes_profissionais", "formacao_e_idiomas",
                        "cargo_atual_old"])
            .drop_columns(["telefone_recado", "infos_basicas", "estado_civil", "skype", "facebook","download_cv", 
                        "email_secundario", "informacoes_profissionais", "informacoes_pessoais", "outro_curso", "formacao_e_idiomas", "cv_en",
                        "email_corporativo", "unidade", "id_ibrati", 'email_superior_imediato', 'nome_superior_imediato', "cargo_atual_old"])
            .drop_duplicate_columns()
            .get_df()

        )
        print(df_applicants_processed)

        ingestor.save_to_parquet(df_applicants_processed, "dados/trusted/applicants.parquet")
    except Exception as e:
        print(f"Erro durante o processamento dos dados de applicants: {e}")
def main():
   
    df_applicants_raw, df_prospects_raw, df_vagas_raw, ingestor = load_data()

    # Verifica se os dados foram carregados corretamente antes de continuar com o processamento
    if df_applicants_raw is not None and df_prospects_raw is not None and df_vagas_raw is not None:
 
        processing_data(df_applicants_raw, df_prospects_raw, df_vagas_raw, ingestor)
    else:
        print("Erro ao carregar os dados. O processamento não pode continuar.")


if __name__ == "__main__":
    main()
