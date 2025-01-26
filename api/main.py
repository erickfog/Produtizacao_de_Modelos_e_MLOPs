import os
import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Inicializar o app Flask
app = Flask(__name__)

# Configurações de diretórios e arquivos
MODEL_DIR = "../models"
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "rl_model.joblib")
S3_MODEL_KEY = "s3_key_to_model.pkl"  # Ajuste para o caminho correto do S3
model = None

# Função para carregar o modelo
def load_model():
    global model
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Supondo que você tenha implementado uma função para baixar do S3
    # s3_service.download_file(S3_MODEL_KEY, LOCAL_MODEL_PATH)
    model = joblib.load(LOCAL_MODEL_PATH)
    return "Modelo carregado com sucesso!"

# Rota para carregar o modelo
@app.route("/load-model", methods=["GET"])
def load_model_route():
    try:
        message = load_model()
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Função para preparar os dados
def prepare_features(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['price (usd)'] = df['Price (USD)']

    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['is_holiday'] = df['Timestamp'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
    df['day_of_month'] = df['Timestamp'].dt.day

    df['moving_avg_7'] = df['price (usd)'].rolling(window=7).mean()
    df['moving_avg_30'] = df['price (usd)'].rolling(window=30).mean()
    df['moving_avg_90'] = df['price (usd)'].rolling(window=90).mean()

    df['daily_return'] = df['price (usd)'].pct_change()
    df['daily_return_abs'] = df['price (usd)'].diff()
    df['volatility'] = df['daily_return'].rolling(window=30).std()

    df['RSI_14'] = calculate_rsi(df['price (usd)'], window=14)

    df['ema_7'] = df['price (usd)'].ewm(span=7, adjust=False).mean()
    df['ema_30'] = df['price (usd)'].ewm(span=30, adjust=False).mean()

    return df.dropna()

# Função para calcular RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Rota para fazer previsões
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Modelo não carregado. Use o endpoint /load-model."}), 400

        # Obter os dados do corpo da requisição
        data = request.get_json()
        df = pd.DataFrame(data)

        # Preparar os dados para o modelo
        features = prepare_features(df)

        # Fazer a previsão
        predictions = model.predict(features)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Executar o servidor Flask
if __name__ == "__main__":
    app.run(debug=True)