from pathlib import Path
import typer
import pandas as pd
from loguru import logger
from tqdm import tqdm
from datetime import datetime

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
    df = df.rename(columns={"Timestamp":"timestamp","Price (USD)":"price (usd)"})

    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day

    # Média móvel de 7 dias (curto prazo)
    df['moving_avg_7'] = df['price (usd)'].rolling(window=7).mean()

    # Média móvel de 30 dias (médio prazo)
    df['moving_avg_30'] = df['price (usd)'].rolling(window=30).mean()

    # Média móvel de 90 dias (longo prazo)
    df['moving_avg_90'] = df['price (usd)'].rolling(window=90).mean()

    # Calcular a variação percentual diária (retorno)
    df['daily_return'] = df['price (usd)'].pct_change()

    # Calcular a variação absoluta diária
    df['daily_return_abs'] = df['price (usd)'].diff()

    df['volatility'] = df['daily_return'].rolling(window=30).std()

    df['RSI_14'] = calculate_rsi(df['price (usd)'], window=14)

    # Média exponencial de 7 dias
    df['ema_7'] = df['price (usd)'].ewm(span=7, adjust=False).mean()

    # Média exponencial de 30 dias
    df['ema_30'] = df['price (usd)'].ewm(span=30, adjust=False).mean()
    
    # Remover NaNs gerados
    return df.dropna()

RAW_DATA_DIR = Path("data/raw/") 
PROCESSED_DATA_DIR = Path("data/processed")

#@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "historical_data_2.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features.parquet",
):
    """
    Calcula as features a partir do dataset e salva o resultado.

    :param input_path: Caminho do dataset de entrada.
    :param output_path: Caminho do arquivo de saída.
    """
    logger.info(f"Lendo dataset de {input_path}...")
    try:
        df = pd.read_parquet(input_path)
        logger.info("Calculando features...")
        df_features = prepare_features(df)

        # Salvar arquivo localmente
        df_features.to_parquet(output_path, index=False)
        logger.info(f"Features salvas temporariamente em {output_path}.")

    except Exception as e:
        logger.error(f"Erro no cálculo de features: {e}")

if __name__ == "__main__":
    main()
