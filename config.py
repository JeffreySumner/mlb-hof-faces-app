"""Configuration values for the HOF face predictor app."""

from pathlib import Path
import os


APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
IMAGE_CACHE_DIR = CACHE_DIR / "images"
PLAYER_INDEX_CACHE = CACHE_DIR / "player_index.csv"
MODEL_CACHE_PATH = CACHE_DIR / "artifacts" / "mlb_hof_model.keras"

# Hugging Face model artifact settings
HOF_MODEL_REPO_ID = os.getenv("HOF_MODEL_REPO_ID", "rpy-ai/mlb-hof-faces")
HOF_MODEL_FILENAME = os.getenv("HOF_MODEL_FILENAME", "artifacts/mlb_hof_model.keras")

# Inference threshold for class=HOF
DEFAULT_HOF_THRESHOLD = float(os.getenv("HOF_THRESHOLD", "0.25"))

# Standard model image size
MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", "128"))

