import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import joblib
import os
from datetime import datetime
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pandas as pd
import numpy as np

import boto3

class S3Services:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_file(self, file_name, s3_key):
        """
        Faz upload de um arquivo para um bucket S3.
        :param file_name: Caminho local do arquivo.
        :param s3_key: Caminho no S3 onde o arquivo será salvo.
        """
        try:
            self.s3_client.upload_file(file_name, self.bucket_name, s3_key)
            print(f"Arquivo {file_name} enviado com sucesso para {s3_key}.")
        except Exception as e:
            print(f"Erro ao enviar o arquivo para o S3: {str(e)}")

    def download_file(self, s3_key, local_file_name):
        """
        Faz download de um arquivo de um bucket S3 para o sistema de arquivos local.
        :param s3_key: Caminho no S3 onde o arquivo está salvo.
        :param local_file_name: Caminho local onde o arquivo será salvo.
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_file_name)
            print(f"Arquivo {s3_key} baixado com sucesso para {local_file_name}.")
        except Exception as e:
            print(f"Erro ao baixar o arquivo do S3: {str(e)}")


# Configurações do S3
bucket_name = "my-firehose-bucket"
s3_model_key = "ml_models/rl_model.pkl"
local_model_path = "../models/rl_model.pkl"

# Inicializar o serviço S3
s3_service = S3Services(bucket_name)

# Função para baixar e carregar o modelo
@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)
    s3_service.download_file(s3_model_key, local_model_path)
    model = joblib.load(local_model_path)
    st.success("Modelo carregado com sucesso!")
    return model



def calculate_rsi(data, window=14):
    """
    Calcula o indicador RSI (Relative Strength Index) de uma série temporal.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    """
    Prepara o dataframe com as features necessárias para o modelo.
    - Adiciona variáveis de tempo, médias móveis, retornos, volatilidade, RSI e EMA.
    - Remove os valores NaN no final.
    """
    # Garantir que as colunas estão no formato correto
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['price (usd)'] = df['Price (USD)']

    # Extrair dia da semana e identificar final de semana
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

    # Identificar feriados específicos (exemplo: 25 de dezembro)
    df['is_holiday'] = df['Timestamp'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)

    # Extrair o número do dia no mês
    df['day_of_month'] = df['Timestamp'].dt.day

    # Médias móveis (curto, médio e longo prazo)
    df['moving_avg_7'] = df['price (usd)'].rolling(window=7).mean()
    df['moving_avg_30'] = df['price (usd)'].rolling(window=30).mean()
    df['moving_avg_90'] = df['price (usd)'].rolling(window=90).mean()

    # Retornos diários
    df['daily_return'] = df['price (usd)'].pct_change()
    df['daily_return_abs'] = df['price (usd)'].diff()

    # Volatilidade (desvio padrão dos retornos diários em janela de 30 dias)
    df['volatility'] = df['daily_return'].rolling(window=30).std()

    # RSI (Relative Strength Index)
    df['RSI_14'] = calculate_rsi(df['price (usd)'], window=14)

    # Médias móveis exponenciais
    df['ema_7'] = df['price (usd)'].ewm(span=7, adjust=False).mean()
    df['ema_30'] = df['price (usd)'].ewm(span=30, adjust=False).mean()

    # Remover NaNs gerados pelas janelas
    return df.dropna()

# Função para buscar dados históricos
def get_historical_data(coin, days=90):
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=["Timestamp", "Price (USD)"])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Erro ao buscar dados históricos: {e}")
        return pd.DataFrame()

# Interface do Streamlit
def main():
    st.title("Previsão de Preço de Bitcoin 🪙")
    st.write("### Utilize dados da API CoinGecko e um modelo treinado para prever o preço do Bitcoin!")

    # Carregar o modelo
    st.info("Carregando modelo do S3...")
    model = load_model()

    # Seleção de criptomoeda
    crypto_coin = st.selectbox("Escolha a criptomoeda:", ["bitcoin"])
    historical_days = st.slider("Quantidade de dias históricos para análise:", 30, 90, 60)
    forecast_days = st.slider("Quantos dias à frente você deseja prever?", 1, 30, 30)

    if st.button("Fazer Previsão"):
        st.info("Buscando dados históricos...")
        data = get_historical_data(crypto_coin, historical_days)

        if not data.empty:
            st.write("### Dados Históricos")
            st.dataframe(data)

            # Preparar as features
            st.info("Preparando features para inferência...")
            prepared_data = prepare_features(data)

            # Fazer previsões para o histórico
            feature_columns = ['day_of_week', 'day_of_month',
                               'moving_avg_7', 'moving_avg_30', 'moving_avg_90']

            # Garantir que as features não tenham NaNs antes da predição
            prepared_data = prepared_data.dropna(subset=feature_columns)

            # Fazer as previsões para os dados históricos
            prepared_data['Predicted Price'] = model.predict(prepared_data[feature_columns])

            # Criar DataFrame para previsões futuras
            future_dates = pd.date_range(start=prepared_data['Timestamp'].iloc[-1] + pd.Timedelta(days=1), 
                                         periods=forecast_days, freq='D')

            # Criar um DataFrame vazio para as previsões futuras
            future_data = pd.DataFrame(future_dates, columns=['Timestamp'])

            # Definir features para os dias futuros
            future_data['day_of_week'] = future_data['Timestamp'].dt.dayofweek
            future_data['is_weekend'] = future_data['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
            future_data['is_holiday'] = future_data['Timestamp'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
            future_data['day_of_month'] = future_data['Timestamp'].dt.day

            # Preencher as features futuras com base nas médias das últimas observações
            future_data['moving_avg_7'] = prepared_data['moving_avg_7'].iloc[-1]
            future_data['moving_avg_30'] = prepared_data['moving_avg_30'].iloc[-1]
            future_data['moving_avg_90'] = prepared_data['moving_avg_90'].iloc[-1]
            future_data['daily_return'] = 0  # Assumindo que o retorno futuro é zero (pode ser ajustado)
            future_data['daily_return_abs'] = 0  # Similar
            future_data['volatility'] = prepared_data['volatility'].iloc[-1]
            future_data['RSI_14'] = prepared_data['RSI_14'].iloc[-1]
            future_data['ema_7'] = prepared_data['ema_7'].iloc[-1]
            future_data['ema_30'] = prepared_data['ema_30'].iloc[-1]

            # Fazer as previsões para os dias futuros
            future_data['Predicted Price'] = model.predict(future_data[feature_columns])

            # Concatenar dados históricos com previsões futuras
            all_data = pd.concat([prepared_data[['Timestamp', 'Price (USD)', 'Predicted Price']],
                                  future_data[['Timestamp', 'Predicted Price']]], ignore_index=True)

            # Plotar a curva de preços reais e previstos
            st.write("### Comparação entre Preço Real e Previsão (Próximos Dias)")
            chart_data = all_data.set_index('Timestamp')
            st.line_chart(chart_data[['Price (USD)', 'Predicted Price']])

            # Exibir a última previsão futura
            last_prediction = future_data['Predicted Price'].iloc[-1]
            st.success(f"🔮 Previsão Final de Preço para o próximo dia: **${last_prediction:.2f} USD**")

        else:
            st.error("Não foi possível obter dados históricos.")




if __name__ == "__main__":
    main()
