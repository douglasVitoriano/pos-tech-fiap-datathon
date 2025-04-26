from DataIngestorGlue import DataIngestor
from DataProcessorGlue import DataProcessor
import logging
import sys

# Configuração de logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    try:
        bucket = 'fiap-datathon-mlet2'
        ingestor = DataIngestor(bucket)

        logger.info("Lendo arquivos JSON da camada landing...")
        df_applicants_raw = ingestor.read_json("landing/applicants.json").T
        df_prospects_raw = ingestor.read_json("landing/prospects.json").T
        df_vagas_raw = ingestor.read_json("landing/vagas.json").T

        logger.info("Validando schema dos arquivos...")
        applicants_schema_raw = {col: str(dtype) for col, dtype in df_applicants_raw.dtypes.items()}
        prospects_schema_raw = {col: str(dtype) for col, dtype in df_prospects_raw.dtypes.items()}
        vagas_schema_raw = {col: str(dtype) for col, dtype in df_vagas_raw.dtypes.items()}

        ingestor.validate_schema(df_applicants_raw, applicants_schema_raw)
        ingestor.validate_schema(df_prospects_raw, prospects_schema_raw)
        ingestor.validate_schema(df_vagas_raw, vagas_schema_raw)

        # Processa vagas
        logger.info("Processando dados de vagas...")
        vagas_processor = DataProcessor(df_vagas_raw)
        df_vagas_processed = (
            vagas_processor
            .reset_index()
            .rename_column({"index": "vaga_id"})
            .json_normalize(['informacoes_basicas', 'perfil_vaga', 'beneficios'])
            .drop_columns(['informacoes_basicas', 'perfil_vaga', 'beneficios'])
            .get_df()
        )
        ingestor.save_to_parquet(df_vagas_processed, "trusted/vagas.parquet")

        # Processa prospects
        logger.info("Processando dados de prospects...")
        prospects_processor = DataProcessor(df_prospects_raw)
        df_prospects_processed = (
            prospects_processor
            .explode_column("prospects")
            .reset_index()
            .json_normalize(["prospects"])
            .rename_column({"index": "codigo_vaga", "codigo": "codigo_applicants", "nome": "nome_applicants"})
            .drop_columns(["prospects"])
            .get_df()
        )
        ingestor.save_to_parquet(df_prospects_processed, "trusted/prospects.parquet")

        # Processa applicants
        logger.info("Processando dados de applicants...")
        applicants_processor = DataProcessor(df_applicants_raw)
        df_applicants_processed = (
            applicants_processor
            .reset_index()
            .rename_column({"index": "codigo_candidato"})
            .json_normalize(["infos_basicas", "informacoes_pessoais", "informacoes_profissionais", "formacao_e_idiomas", "cargo_atual"])
            .drop_columns([
                "telefone_recado", "infos_basicas", "estado_civil", "skype", "facebook", "download_cv",
                "email_secundario", "informacoes_profissionais", "informacoes_pessoais", "outro_curso",
                "formacao_e_idiomas", "cv_en", "email_corporativo", "unidade", "id_ibrati",
                "email_superior_imediato", "nome_superior_imediato"
            ])
            .drop_duplicate_columns()
            .get_df()
        )
        ingestor.save_to_parquet(df_applicants_processed, "trusted/applicants.parquet")

        logger.info("Processamento concluído com sucesso.")

    except Exception as e:
        logger.exception(f"Erro no job Glue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
