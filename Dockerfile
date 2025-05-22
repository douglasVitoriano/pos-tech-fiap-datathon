# Imagem base
FROM python:3.10-slim

# Define diretório de trabalho
WORKDIR /src

# Copia arquivos para o container
COPY . .

# Instala dependências
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expondo porta padrão FastAPI
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "src.api_ml_job_matching:app", "--host", "0.0.0.0", "--port", "8000"]