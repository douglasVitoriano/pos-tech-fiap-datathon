import pandas as pd
import boto3
import json
import io

class DataIngestor:
    def __init__(self, bucket_name):
        self.bucket = bucket_name
        self.s3_client = boto3.client('s3')

    def read_json(self, s3_key):
        """
        Lê um arquivo JSON do S3 e retorna um DataFrame.
        """
        response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        return pd.DataFrame(data)
    
    def read_parquet(self, s3_key):
        """
        Lê um arquivo Parquet do S3 e retorna um DataFrame.
        """
        response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
        # Obtém o conteúdo do arquivo
        content = response['Body'].read()
        
        # Lê o conteúdo Parquet diretamente de bytes usando pandas
        return pd.read_parquet(io.BytesIO(content))
    
    def validate_schema(self, df, expected_schema):
        """
        expected_schema: dict com nome da coluna e tipo esperado, ex: {"id": "int64", "nome": "object"}
        """
        actual_schema = df.dtypes.to_dict()
        for col, expected_type in expected_schema.items():
            if col not in df.columns:
                raise ValueError(f"Coluna ausente: {col}")
            if str(actual_schema[col]) != expected_type:
                raise TypeError(f"Tipo incorreto para {col}: esperado {expected_type}, encontrado {actual_schema[col]}")
            
    def save_to_parquet(self, df, path, index=False):
        """
        Salva o DataFrame como um arquivo Parquet diretamente no S3 usando s3fs.

        Parameters:
        - df (DataFrame): O DataFrame a ser salvo.
        - path (str): O caminho do arquivo no S3, incluindo o bucket e o prefixo
        - index (bool): Se o índice deve ser incluído como coluna no Parquet.
        """
        # Usando o s3fs para salvar diretamente no S3
        parquet_path = f"{self.bucket}/{path}"

        # Salvando o arquivo Parquet no S3 diretamente
        df.to_parquet(f"s3://{parquet_path}", engine="pyarrow", index=index)

        print(f"Arquivo Parquet salvo com sucesso em: s3://{self.bucket}/{path}")
