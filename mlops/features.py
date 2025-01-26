from pathlib import Path
import typer
import pandas as pd
from loguru import logger
from tqdm import tqdm
from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime


app = typer.Typer()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcula o índice de força relativa (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o dataframe com as features necessárias para o modelo.
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['price (usd)'] = df['Price (USD)']

    # Features de tempo
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['is_holiday'] = df['Timestamp'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
    df['day_of_month'] = df['Timestamp'].dt.day

    # Médias móveis
    df['moving_avg_7'] = df['price (usd)'].rolling(window=7).mean()
    df['moving_avg_30'] = df['price (usd)'].rolling(window=30).mean()
    df['moving_avg_90'] = df['price (usd)'].rolling(window=90).mean()

    # Retornos e volatilidade
    df['daily_return'] = df['price (usd)'].pct_change()
    df['daily_return_abs'] = df['price (usd)'].diff()
    df['volatility'] = df['daily_return'].rolling(window=30).std()

    # RSI e EMA
    df['RSI_14'] = calculate_rsi(df['price (usd)'], window=14)
    df['ema_7'] = df['price (usd)'].ewm(span=7, adjust=False).mean()
    df['ema_30'] = df['price (usd)'].ewm(span=30, adjust=False).mean()

    # Remover NaNs gerados pelas janelas
    return df.dropna()


def upload_to_s3(file_path: Path, bucket: str, s3_key: str):
    """
    Faz upload de um arquivo local para um bucket S3.

    :param file_path: Caminho local do arquivo.
    :param bucket: Nome do bucket S3.
    :param s3_key: Caminho do arquivo no bucket S3.
    """
    s3 = boto3.client("s3")
    try:
        s3.upload_file(str(file_path), bucket, s3_key)
        logger.success(f"Arquivo enviado para o S3: s3://{bucket}/{s3_key}")
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Erro ao enviar arquivo para o S3: {e}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    bucket: str = "meu-bucket",
    base_s3_key: str = "processed/crypto_features",
):
    """
    Calcula as features a partir do dataset e envia para o S3.

    :param input_path: Caminho do dataset de entrada.
    :param output_path: Caminho do arquivo de saída.
    :param bucket: Nome do bucket S3.
    :param base_s3_key: Caminho base no S3 para salvar as features.
    """
    logger.info(f"Lendo dataset de {input_path}...")
    try:
        df = pd.read_csv(input_path)
        logger.info("Calculando features...")
        df_features = prepare_features(df)

        # Salvar arquivo localmente
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)
        logger.info(f"Features salvas temporariamente em {output_path}.")

        # Gerar partições com base na data mais recente
        latest_date = df_features['Timestamp'].max()
        year, month, day = latest_date.year, latest_date.month, latest_date.day
        s3_key = f"{base_s3_key}/year={year}/month={month:02}/day={day:02}/features.csv"

        # Enviar para o S3
        upload_to_s3(output_path, bucket, s3_key)

    except Exception as e:
        logger.error(f"Erro no cálculo de features: {e}")


if __name__ == "__main__":
    app()