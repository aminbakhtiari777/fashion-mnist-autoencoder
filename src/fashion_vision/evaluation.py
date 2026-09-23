"""Framework-independent evaluation shared by all experiment tracks."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def _image_batch(values: np.ndarray, name: str) -> np.ndarray:
    batch = np.asarray(values, dtype=np.float64)
    if batch.ndim not in {3, 4} or batch.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty image batch")
    if not np.isfinite(batch).all():
        raise ValueError(f"{name} contains non-finite values")
    return batch


def classification_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or labels.ndim != 1:
        raise ValueError("labels must be 1D and probabilities must be 2D")
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must contain the same samples")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    predictions = probabilities.argmax(axis=1)
    classes = np.arange(probabilities.shape[1])
    names = class_names or [str(value) for value in classes]
    if len(names) != len(classes):
        raise ValueError("class_names length must match probability columns")
    recalls = recall_score(labels, predictions, labels=classes, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes,
                                   average="macro", zero_division=0)),
        "per_class_recall": {
            name: float(value) for name, value in zip(names, recalls)
        },
    }


def reconstruction_summary(
    original: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float = 1.0,
) -> dict[str, float]:
    original = _image_batch(original, "original")
    reconstructed = _image_batch(reconstructed, "reconstructed")
    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed shapes must match")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    error = original - reconstructed
    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))
    psnr = math.inf if mse == 0 else float(10 * math.log10((data_range**2) / mse))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def generation_summary(images: np.ndarray, maximum_samples: int = 256) -> dict[str, float]:
    """Return lightweight collapse indicators without claiming perceptual quality."""
    images = _image_batch(images, "images")
    if maximum_samples < 2:
        raise ValueError("maximum_samples must be at least 2")
    selected = images[:maximum_samples].reshape(min(len(images), maximum_samples), -1)
    if len(selected) < 2:
        diversity = 0.0
    else:
        distances = np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=2)
        diversity = float(distances[np.triu_indices(len(selected), k=1)].mean())
    return {
        "sample_diversity": diversity,
        "pixel_mean": float(images.mean()),
        "pixel_std": float(images.std()),
        "pixel_min": float(images.min()),
        "pixel_max": float(images.max()),
    }
