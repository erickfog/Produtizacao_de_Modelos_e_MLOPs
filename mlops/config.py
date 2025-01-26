import os
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Consumir variáveis do .env
PROJ_ROOT = os.getenv("PROJ_ROOT", Path(__file__).resolve().parents[1])  # Fallback para caminho local se não definido
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = os.getenv("DATA_DIR", f"{PROJ_ROOT}/data")
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", f"{DATA_DIR}/raw")
INTERIM_DATA_DIR = os.getenv("INTERIM_DATA_DIR", f"{DATA_DIR}/interim")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", f"{DATA_DIR}/processed")
EXTERNAL_DATA_DIR = os.getenv("EXTERNAL_DATA_DIR", f"{DATA_DIR}/external")

MODELS_DIR = os.getenv("MODELS_DIR", f"{PROJ_ROOT}/models")

REPORTS_DIR = os.getenv("REPORTS_DIR", f"{PROJ_ROOT}/reports")
FIGURES_DIR = os.getenv("FIGURES_DIR", f"{REPORTS_DIR}/figures")

# Log os caminhos para verificação
logger.info(f"DATA_DIR: {DATA_DIR}")
logger.info(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
logger.info(f"MODELS_DIR: {MODELS_DIR}")

# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass