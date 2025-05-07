import pandas as pd
import os

class DataIngestor:
    def __init__(self):
        pass

    def read_json(self, path, columns=None, dtype=None, orient=None):
        """
        Lê um arquivo JSON e retorna um DataFrame pandas.

        :param path: Caminho do arquivo JSON
        :param columns: Lista de colunas a serem retornadas (opcional)
        :param dtype: Dicionário com os tipos das colunas (opcional)
        :param orient: Forma de estruturação dos dados no JSON (padrão: auto)
        :return: DataFrame pandas
        """
        df = pd.read_json(path, orient=orient, dtype=dtype)

        if columns:
            df = df[columns]

        return df
    
    def read_parquet(self, filename):
        """
        Lê um arquivo Parquet específico de um diretório local e retorna um DataFrame.
        """
        # Cria o caminho completo para o arquivo
        file_path = os.path.join(self.directory, filename)
        
        # Verifica se o arquivo existe
        if os.path.exists(file_path) and file_path.endswith('.parquet'):
            # Lê o arquivo Parquet
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"O arquivo {filename} não foi encontrado no diretório {self.directory} ou não é um arquivo Parquet válido.")
    
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
        df.to_parquet(path, index=index)
