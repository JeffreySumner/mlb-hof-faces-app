"""Model loading utilities with Hugging Face artifact caching."""

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

from config import (
    APP_ROOT,
    HOF_MODEL_FILENAME,
    HOF_MODEL_REPO_ID,
    MODEL_CACHE_PATH,
)


def ensure_model_file() -> Path:
    """Ensure a local Keras model file exists and return its path."""
    if MODEL_CACHE_PATH.exists():
        return MODEL_CACHE_PATH

    # Prefer bundled local model if available in repo.
    bundled_candidates = [
        APP_ROOT / "model" / "mlb_hof_model.keras",
        APP_ROOT / "model" / "hof_model.keras",
    ]
    bundled_model = next((p for p in bundled_candidates if p.exists()), None)
    if bundled_model is not None:
        MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled_model, MODEL_CACHE_PATH)
        return MODEL_CACHE_PATH

    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Try configured filename first, then common fallback paths.
    candidate_filenames = [
        HOF_MODEL_FILENAME,
        "artifacts/mlb_hof_model.keras",
        "model/mlb_hof_model.keras",
        "model/hof_model.keras",
        "mlb_hof_model.keras",
        "hof_model.keras",
    ]
    tried = []
    for filename in candidate_filenames:
        if filename in tried:
            continue
        tried.append(filename)
        try:
            downloaded = hf_hub_download(
                repo_id=HOF_MODEL_REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(MODEL_CACHE_PATH.parent),
                local_dir_use_symlinks=False,
            )
            return Path(downloaded)
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not locate model artifact locally or on Hugging Face. "
        f"Tried repo={HOF_MODEL_REPO_ID} with filenames={tried}. "
        "Either upload the model artifact or set HOF_MODEL_FILENAME to a valid path."
    )


def load_model():
    """Load and return the TensorFlow model."""
    import tensorflow as tf

    model_path = ensure_model_file()
    return tf.keras.models.load_model(model_path)

