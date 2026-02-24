"""Prediction helpers for HOF model outputs."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    prob_not_hof: float
    prob_hof: float
    threshold: float
    label: str


def predict_hof_probability(model, model_input: np.ndarray, threshold: float) -> PredictionResult:
    """Run model inference and convert to friendly prediction output."""
    expected_shape = model.input_shape
    # expected_shape examples:
    # dense legacy: (None, 1024)
    # cnn modern:   (None, 128, 128, 1)
    if len(expected_shape) == 2:
        infer_input = model_input.reshape(1, -1)
    elif len(expected_shape) == 4:
        infer_input = model_input.reshape(1, model_input.shape[0], model_input.shape[1], 1)
    else:
        raise ValueError(f"Unsupported model input shape: {expected_shape}")

    pred = model.predict(infer_input, verbose=0)[0]
    prob_not_hof = float(pred[0])
    prob_hof = float(pred[1])
    label = "HOF" if prob_hof >= threshold else "Not HOF"
    return PredictionResult(
        prob_not_hof=prob_not_hof,
        prob_hof=prob_hof,
        threshold=threshold,
        label=label,
    )

