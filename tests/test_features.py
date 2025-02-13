import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Importe as funções que você quer testar
from mlops.features import prepare_features  # ajuste o caminho conforme necessário

def test_prepare_features():
    # Cria uma sequência de datas com 100 entradas (garantindo que as operações de rolling terão dados suficientes)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Cria uma série de preços que varia de 100 a 200 (exemplo simples: linha reta)
    prices = np.linspace(100, 200, 100)
    
    # Cria o dataframe de teste com as colunas originais esperadas
    df_input = pd.DataFrame({
        "Timestamp": dates,
        "Price (USD)": prices
    })
    
    # Converte a coluna 'Timestamp' para datetime, caso não esteja
    df_input["Timestamp"] = pd.to_datetime(df_input["Timestamp"])
    
    # Aplica a função para preparar as features
    df_features = prepare_features(df_input)
    
    # Lista das colunas esperadas após o processamento
    expected_columns = [
        "timestamp", "price (usd)", "day_of_week", "day_of_month",
        "moving_avg_7", "moving_avg_30", "moving_avg_90", "daily_return",
        "daily_return_abs", "volatility", "RSI_14", "ema_7", "ema_30"
    ]
    
    # Verifica se todas as colunas esperadas estão presentes
    for col in expected_columns:
        assert col in df_features.columns, f"A coluna {col} não foi encontrada no dataframe de features."
    
    # Verifica se não há valores nulos (pois a função aplica dropna())
    assert df_features.isnull().sum().sum() == 0, "Existem valores nulos no dataframe de features."
    
    # Verifica se os valores de RSI estão entre 0 e 100
    assert df_features["RSI_14"].between(0, 100).all(), "Os valores de RSI_14 não estão no intervalo de 0 a 100."

    # Opcional: Verifica se o número de linhas é coerente com a aplicação do dropna()
    # Por exemplo, se você tem 100 linhas e usa médias móveis de 90 dias, o resultado pode ter menos linhas.
    assert len(df_features) < 100, "O número de linhas não diminuiu após a aplicação do dropna(), o que pode indicar que as janelas de rolling não foram aplicadas corretamente."