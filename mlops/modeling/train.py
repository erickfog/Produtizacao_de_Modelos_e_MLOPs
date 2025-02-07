from pathlib import Path
import typer
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

#from mlops.config import MODELS_DIR, PROCESSED_DATA_DIR

PROCESSED_DATA_DIR = Path('data/processed/')
MODELS_DIR = Path('models/')

app = typer.Typer()

@app.command()
def train_models(
    features_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """
    Treina modelos de regressão utilizando variáveis derivadas.
    """
    logger.info("Carregando dados...")
    df_features = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(features_path)
    
    logger.info("DADOS CARREGADOS")
    # Lista das variáveis derivadas
    derived_columns = ['day_of_week', 'day_of_month', 'moving_avg_7', 'moving_avg_30', 'moving_avg_90']
    
    X = df_features[derived_columns]
    y = df_features['price (usd)']
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
            "LinearRegression": LinearRegression(),
            "XGBRegressor": XGBRegressor(n_estimators=100, learning_rate=0.4, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, random_state=42)
        }

    best_model = None
    best_mae = float("inf")
        
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        if mae < best_mae:
            best_mae = mae
            best_model = model

    joblib.dump(best_model, model_path)
    logger.success(f"Melhor modelo ({best_model.__class__.__name__}) salvo em {model_path} com MAE {best_mae}.")

    # Exibir métricas
    #logger.success(f'Regressão Linear - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    #logger.success(f'XGBoost - RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}, R²: {xgb_r2:.4f}')
    #logger.success(f'Gradient Boosting - RMSE: {gb_rmse:.4f}, MAE: {gb_mae:.4f}, R²: {gb_r2:.4f}')

if __name__ == "__main__":
    app()
