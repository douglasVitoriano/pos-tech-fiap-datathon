import pandas as pd

class DataProcessor:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def reset_index(self):
        """
        Reseta o índice do DataFrame armazenado em self.df in-place.
        O índice antigo será adicionado como uma coluna no DataFrame.
        """
        self.df.reset_index(inplace=True)
        return self
    
    def rename_column(self, columns_map):
        """
        Renomeia uma coluna no DataFrame armazenado em self.df.

        Parâmetros:
        - old_name (str): nome atual da coluna
        - new_name (str): novo nome da coluna

        Retorna:
        - DataFrame com a coluna renomeada
        """
        self.df.rename(columns=columns_map, inplace=True)
        return self
    
    def json_normalize(self, columns):
        """
            Normaliza várias colunas JSON de um DataFrame.

        Parâmetros:
            - columns (list): lista de nomes de colunas a serem normalizadas.

        Retorna:
            - self
        """
        for column in columns:
            if column in self.df.columns:
                # Converte cada linha da coluna em dict (ou None)
                data_to_normalize = self.df[column].apply(lambda x: x if isinstance(x, dict) else {})
                # Normaliza 
                normalized_df = pd.json_normalize(data_to_normalize)
                # Concatena mantendo o index
                self.df = pd.concat([self.df, normalized_df], axis=1)
        return self


    
    def explode_column(self, column):
        """
        Aplica o explode em uma coluna do DataFrame armazenado em self.df.

        Parâmetros:
        - column (str): nome da coluna que contém listas e que será explodida em várias linhas.

        Retorna:
        - self (para permitir encadeamento de métodos)
        """
        self.df = self.df.explode(column)
        return self

    def normalize_column(self, column, lowercase=True, strip=True):
        if lowercase:
            self.df[column] = self.df[column].str.lower()
        if strip:
            self.df[column] = self.df[column].str.strip()
        return self


    def add_column(self, column_name, func):
        """
        Adiciona nova coluna usando função sobre o DataFrame
        """
        self.df[column_name] = self.df.apply(func, axis=1)
        return self.df
    
    def drop_columns(self, columns):
        """
        Remove uma ou mais colunas do DataFrame armazenado em self.df.

        Parâmetros:
        - columns (list): lista com os nomes das colunas a serem removidas.

        Retorna:
        - self 
        """
        if not isinstance(columns, list):
            raise TypeError("O argumento 'columns' deve ser uma lista de nomes de colunas.")

        self.df.drop(columns=columns, inplace=True)
        return self

    
    def clean_nulls(self, strategy='drop', fill_value=None):
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            self.df = self.df.fillna(fill_value)
        return self

    def drop_duplicates(self, subset=None):
        self.df = self.df.drop_duplicates(subset=subset)
        return self
    
    def convert_types(self, type_map):
        """
        Força conversão de tipos com tratamento de erro
        """
        for col, t in type_map.items():
            try:
                self.df[col] = self.df[col].astype(t)
            except Exception as e:
                print(f"Erro convertendo coluna {col} para {t}: {e}")
        return self
    
    def group_and_aggregate(self, group_cols, agg_dict):
        self.df = self.df.groupby(group_cols).agg(agg_dict).reset_index()
        return self
    
    def get_columns(self):
        return self.df.columns.tolist()

    def get_df(self):
        return self.df
    
