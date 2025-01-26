from pathlib import Path
import typer
import requests
import pandas as pd
from datetime import datetime
from loguru import logger
from tqdm import tqdm
from mlops.config import RAW_DATA_DIR
import boto3
from botocore.exceptions import BotoCoreError, ClientError

app = typer.Typer()


def get_historical_data(coin: str, days: int = 90) -> pd.DataFrame:
    """
    Busca dados históricos de preços de uma criptomoeda usando a API CoinGecko.

    :param coin: Nome da moeda (ex: 'bitcoin', 'ethereum').
    :param days: Número de dias para buscar dados.
    :return: DataFrame com os dados históricos.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=["Timestamp", "Price (USD)"])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        logger.success(f"Dados históricos para {coin} obtidos com sucesso!")
        return df
    except Exception as e:
        logger.error(f"Erro ao buscar dados históricos: {e}")
        return pd.DataFrame()


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
def fetch_and_upload_data(
    coin: str,
    days: int = 90,
    bucket: str = "meu-bucket",
    base_s3_key: str = "raw/crypto_data",
):
    """
    Busca dados históricos de uma criptomoeda e envia para o S3 com partições.

    :param coin: Nome da moeda (ex: 'bitcoin', 'ethereum').
    :param days: Número de dias para buscar dados.
    :param bucket: Nome do bucket S3.
    :param base_s3_key: Caminho base no S3 para os arquivos.
    """
    logger.info(f"Buscando dados históricos para {coin} nos últimos {days} dias...")
    df = get_historical_data(coin, days)

    if not df.empty:
        # Obter a data mais recente dos dados para criar partições
        latest_date = df['Timestamp'].max()
        year, month, day = latest_date.year, latest_date.month, latest_date.day

        # Criar caminho S3 com partições
        s3_key = f"{base_s3_key}/year={year}/month={month:02}/day={day:02}/crypto_data.csv"

        # Salvar localmente em um arquivo temporário
        local_file_path = RAW_DATA_DIR / "crypto_data.csv"
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_file_path, index=False)
        logger.info(f"Arquivo salvo temporariamente em {local_file_path}.")

        # Enviar para o S3
        upload_to_s3(local_file_path, bucket, s3_key)
    else:
        logger.warning("Nenhum dado foi salvo devido a erro ou resposta vazia.")


if __name__ == "__main__":
    app()