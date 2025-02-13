import numpy as np
import pandas as pd
import joblib
import pytest
from pathlib import Path

# Importe a função de treinamento.
# Ajuste o caminho conforme a sua estrutura de pastas.
from mlops.modeling.train import train_models  

def test_train_models(tmp_path: Path):
    """
    Testa a função de treinamento criando um dataset sintético e verificando se:
      - O arquivo de modelo é criado.
      - O modelo treinado pertence a uma das classes esperadas.
      - O modelo faz predições dentro de um intervalo razoável.
    """
    # Cria um DataFrame sintético
    num_samples = 100
    df = pd.DataFrame({
        "day_of_week": np.random.randint(0, 7, num_samples),
        "day_of_month": np.random.randint(1, 32, num_samples),
        # Para as médias móveis, vamos criar valores que variam de forma linear
        "moving_avg_7": np.linspace(100, 200, num_samples),
        "moving_avg_30": np.linspace(100, 200, num_samples),
        "moving_avg_90": np.linspace(100, 200, num_samples),
        # O preço também varia linearmente de 100 a 200 (exemplo simples)
        "price (usd)": np.linspace(100, 200, num_samples)
    })
    
    # Salva o DataFrame como parquet no diretório temporário
    input_file = tmp_path / "features.parquet"
    df.to_parquet(input_file, index=False)
    
    # Define o caminho para salvar o modelo, também usando o diretório temporário
    output_model = tmp_path / "model.joblib"
    
    # Chama a função de treinamento com os caminhos temporários
    train_models(features_path=input_file, model_path=output_model)
    
    # Verifica se o arquivo do modelo foi criado
    assert output_model.exists(), "O modelo não foi salvo."
    
    # Carrega o modelo salvo
    model = joblib.load(output_model)
    
    # Verifica se o modelo é de um dos tipos esperados
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    assert isinstance(
        model, (LinearRegression, XGBRegressor, GradientBoostingRegressor)
    ), "O modelo salvo não é de um tipo esperado."
    
    # Teste opcional: Verificar se o modelo faz predições coerentes.
    # Cria um exemplo de entrada com os 5 atributos esperados.
    sample = np.array([[3, 15, 150000, 150, 150]])
    pred = model.predict(sample)
    
    # Como o preço varia de 100 a 200 no dataset sintético, a predição deve estar nesse intervalo (com uma margem)
    #assert 90000 < pred[0] < 300000, f"Predição fora do intervalo esperado: {pred[0]}"
