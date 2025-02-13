# Usar a imagem oficial do Python como base
FROM python:3.10.12-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar o arquivo de dependências (requirements.txt) para dentro do container
COPY requirements.txt /app/

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código da pasta api para dentro do container
COPY api/ /app/api/

# Definir a URL do servidor MLflow (se estiver rodando fora do container)
ENV MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

# Expor a porta que o Flask vai rodar (porta 5001)
EXPOSE 5001

# Comando para rodar a API
CMD ["python", "/app/api/main.py"]