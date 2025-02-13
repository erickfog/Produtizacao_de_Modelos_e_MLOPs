import os
import requests
import pandas as pd
import mlflow.pyfunc
from flask import Flask, jsonify

# Inicializar o app Flask
app = Flask(__name__)

# Configuração do MLflow Registry
MLFLOW_MODEL_NAME = "bitcoin_price_predict"
MLFLOW_MODEL_STAGE = "champion"

# Carregar o modelo do MLflow Registry
model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_STAGE}")

# Função para buscar dados históricos do CoinGecko
def get_historical_data(coin="bitcoin", days=90):
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=["Timestamp", "Price (USD)"])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        return df
    except Exception:
        return pd.DataFrame()

# Função para calcular RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Função para preparar os dados
def prepare_features(df):
    df['price (usd)'] = df['Price (USD)']
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['is_holiday'] = df['Timestamp'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
    df['day_of_month'] = df['Timestamp'].dt.day
    df['moving_avg_7'] = df['price (usd)'].rolling(window=7).mean()
    df['moving_avg_30'] = df['price (usd)'].rolling(window=30).mean()
    df['moving_avg_90'] = df['price (usd)'].rolling(window=90).mean()
    df['daily_return'] = df['price (usd)'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=30).std()
    df['RSI_14'] = calculate_rsi(df['price (usd)'], window=14)
    df['ema_7'] = df['price (usd)'].ewm(span=7, adjust=False).mean()
    df['ema_30'] = df['price (usd)'].ewm(span=30, adjust=False).mean()
    
    return df.dropna()

# Rota para previsão sem necessidade de payload
@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Buscar os últimos 90 dias de dados históricos
        df_historical = get_historical_data()

        # Verificar se os dados foram recuperados
        if df_historical.empty:
            return jsonify({"error": "Erro ao buscar dados históricos"}), 400

        # Preparar as features
        features = prepare_features(df_historical)

        # Verificar se há dados suficientes para a previsão
        if features.empty:
            return jsonify({"error": "Dados insuficientes para previsão"}), 400

        # Criar o próximo timestamp (previsão para o dia seguinte)
        next_timestamp = df_historical["Timestamp"].max() + pd.Timedelta(minutes=1)

        # Criar um DataFrame para previsão com as últimas features conhecidas
        future_features = features.iloc[-1:].copy()
        future_features["Timestamp"] = next_timestamp  # Ajustar o timestamp futuro

        # Fazer a previsão
        prediction = model.predict(future_features.drop(columns=["Timestamp"]))[0]

        return jsonify({
            "next_date": next_timestamp.strftime("%Y-%m-%d"),
            "predicted_price": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Executar o servidor Flask
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Rodar na porta 5002 para evitar conflito com o MLflow
