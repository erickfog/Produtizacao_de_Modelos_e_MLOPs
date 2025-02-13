from pathlib import Path
import typer
import pandas as pd
import joblib
import requests
from loguru import logger
from datetime import datetime


def predict(
    model_path: Path = Path("models/rl_model.joblib"),
    output_path: Path = Path("data/processed/predictions.parquet"),
):
    """
    Faz previsões utilizando o melhor modelo treinado.
    """
    try:
        # Carregar modelo treinado
        logger.info("Carregando modelo...")
        model = joblib.load(model_path)

        # Criar entrada de previsão
        timestamp = datetime.now()
        X_new = pd.DataFrame({
            "day_of_week": [timestamp.weekday()],
            "day_of_month": [timestamp.day],
            "moving_avg_7": [94472.382736],
            "moving_avg_30": [94472.382736],
            "moving_avg_90": [94378.144222]
        })

        # Fazer previsão
        prediction = model.predict(X_new)
        logger.info(f"Preço previsto: {prediction[0]:.2f}")

        # Salvar previsão
        df_prediction = pd.DataFrame({
            "timestamp": [timestamp],
            "predicted_price": prediction
        })
        df_prediction.to_parquet(output_path, index=False)
        logger.success(f"Previsões salvas em {output_path}.")

    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {e}")

if __name__ == "__main__":
    predict()
