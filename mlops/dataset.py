from pathlib import Path
import typer
import requests
import pandas as pd
from datetime import datetime
from loguru import logger
from tqdm import tqdm
#from mlops.config import RAW_DATA_DIR
import boto3

RAW_DATA_DIR = "data/raw/" 



#app = typer.Typer()


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




#@app.command()
def fetch_and_upload_data(
    coin: str = 'bitcoin',
    days: int = 90,
):
    """
    Busca dados históricos de uma criptomoeda e envia para o S3 com partições.

    :param coin: Nome da moeda (ex: 'bitcoin', 'ethereum').
    :param days: Número de dias para buscar dados.
    """
    logger.info(f"Buscando dados históricos para {coin} nos últimos {days} dias...")
    df = get_historical_data(coin, days)

    if not df.empty:
        # Obter a data mais recente dos dados para criar partições        
        # Salvar localmente em um arquivo temporário
        local_file_path = Path(RAW_DATA_DIR) / 'historical_data_2.parquet'
        df.to_parquet(local_file_path, index=False)
        logger.info(f"Arquivo salvo temporariamente em {local_file_path}.")

    else:
        logger.warning("Nenhum dado foi salvo devido a erro ou resposta vazia.")


if __name__ == "__main__":
    fetch_and_upload_data()