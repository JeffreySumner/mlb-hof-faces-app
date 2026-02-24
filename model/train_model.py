"""Train a 128x128 CNN for HOF face prediction."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from tensorflow import keras
from tensorflow.keras import layers

import sys

# Add parent directory to path for project modules
sys.path.append(str(Path(__file__).parent.parent))

from inference.preprocess import preprocess_for_model


RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


def ensure_dataset(data_dir: Path, repo_id: str) -> Path:
    """Ensure local training dataset exists, otherwise download from HF."""
    hof_dir = data_dir / "hof"
    nothof_dir = data_dir / "not-hof"
    if hof_dir.exists() and nothof_dir.exists():
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=["hof/*", "not-hof/*", "metadata.csv"],
    )
    return data_dir


def load_labeled_paths(data_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Load image paths and binary labels."""
    hof_files = sorted((data_dir / "hof").glob("*.jpg"))
    nothof_files = sorted((data_dir / "not-hof").glob("*.jpg"))
    files = hof_files + nothof_files
    labels = np.array([1] * len(hof_files) + [0] * len(nothof_files), dtype=np.int32)
    if len(files) == 0:
        raise ValueError("No training images found in dataset.")
    print(f"Loaded {len(hof_files)} HOF and {len(nothof_files)} not-HOF images")
    return files, labels


def _load_image_tensor(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr_2d, _, _ = preprocess_for_model(img, size=image_size)
    return arr_2d[..., np.newaxis]


def build_arrays(files: list[Path], labels: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([_load_image_tensor(f, image_size=image_size) for f in files], dtype=np.float32)
    y = keras.utils.to_categorical(labels, num_classes=2)
    return x, y


def stratified_split(files: list[Path], labels: np.ndarray, test_ratio: float = 0.25):
    """Simple deterministic stratified split without sklearn."""
    idx_hof = np.where(labels == 1)[0]
    idx_not = np.where(labels == 0)[0]
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(idx_hof)
    rng.shuffle(idx_not)

    n_hof_test = max(1, int(len(idx_hof) * test_ratio))
    n_not_test = max(1, int(len(idx_not) * test_ratio))

    test_idx = np.concatenate([idx_hof[:n_hof_test], idx_not[:n_not_test]])
    train_idx = np.concatenate([idx_hof[n_hof_test:], idx_not[n_not_test:]])

    train_files = [files[i] for i in train_idx]
    test_files = [files[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    return train_files, train_labels, test_files, test_labels


def build_model(image_size: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(image_size, image_size, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train 128x128 HOF face model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/training/mlb-hof-faces"))
    parser.add_argument("--repo-id", type=str, default="rpy-ai/mlb-hof-faces")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("Preparing dataset...")
    data_dir = ensure_dataset(args.data_dir, repo_id=args.repo_id)
    files, labels = load_labeled_paths(data_dir)
    train_files, train_labels, test_files, test_labels = stratified_split(files, labels, test_ratio=0.25)
    print(f"Train size: {len(train_files)} | Test size: {len(test_files)}")

    print("Building arrays...")
    train_x, train_y = build_arrays(train_files, train_labels, image_size=args.image_size)
    test_x, test_y = build_arrays(test_files, test_labels, image_size=args.image_size)

    print("Building model...")
    model = build_model(image_size=args.image_size)
    model.summary()

    # class weighting for imbalance
    n_hof = int(train_labels.sum())
    n_not = int(len(train_labels) - n_hof)
    total = len(train_labels)
    class_weight = {0: total / (2 * max(1, n_not)), 1: total / (2 * max(1, n_hof))}
    print(f"Class weights: {class_weight}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    _ = history

    print("Evaluating...")
    metrics = model.evaluate(test_x, test_y, verbose=0, return_dict=True)
    print("Test metrics:", metrics)

    out_path = Path(__file__).parent / "mlb_hof_model.keras"
    model.save(out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
