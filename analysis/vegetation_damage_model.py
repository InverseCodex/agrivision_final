from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

EPSILON = 1e-6
DEFAULT_MAX_DIMENSION = 1024

VEGETATION_DAMAGE_MODEL_DATA: dict[str, Any] = {
    "feature_names": ["tgi", "std_g", "stand_uniformity_score"],
    "mean": [11.493968089808142, 0.20776145396004106, 0.5479587696340675],
    "std": [2.994313774600539, 0.031041996343169818, 0.20075792893869424],
    "weights": [-3.572818249739656, -3.155527625428556, -1.52346795298902, 0.36292581024015347],
    "threshold": 0.5,
    "positive_class": "unhealthy_damaged",
}

VEGETATION_MATURITY_MODEL_DATA: dict[str, Any] = {
    "feature_names": ["tgi", "stand_uniformity_score"],
    "mean": [11.493968089808142, 0.5479587696340675],
    "std": [2.994313774600539, 0.20075792893869424],
    "weights": [-3.0982337062798933, 4.724495119287652, -3.8518273607142386],
    "threshold": 0.7599999999999999,
    "positive_class": "healthy_mature",
}


@dataclass(frozen=True)
class VegetationDamagePrediction:
    predicted_label: str
    probability: float
    confidence: float
    healthy_probability: float
    unhealthy_damaged_probability: float
    threshold: float
    feature_values: dict[str, float]
    feature_names: list[str]


@dataclass(frozen=True)
class VegetationMaturityPrediction:
    predicted_label: str
    probability: float
    confidence: float
    maturity_probability: float
    threshold: float
    feature_values: dict[str, float]
    feature_names: list[str]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _resize_for_features(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return image

    scale = max_dimension / float(longest_side)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _compute_patch_uniformity(mask: np.ndarray, patch_rows: int = 8, patch_cols: int = 8) -> float:
    grid = cv2.resize(mask.astype(np.float32), (patch_cols, patch_rows), interpolation=cv2.INTER_AREA)
    mean_value = float(np.mean(grid))
    std_value = float(np.std(grid))
    if mean_value <= EPSILON:
        return 0.0

    coefficient_of_variation = std_value / (mean_value + EPSILON)
    return float(np.clip(1.0 - coefficient_of_variation, 0.0, 1.0))


def _normalize_model(model_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": [str(item) for item in model_data["feature_names"]],
        "mean": np.asarray(model_data["mean"], dtype=np.float64),
        "std": np.where(
            np.asarray(model_data["std"], dtype=np.float64) < 1e-8,
            1.0,
            np.asarray(model_data["std"], dtype=np.float64),
        ),
        "weights": np.asarray(model_data["weights"], dtype=np.float64),
        "threshold": float(model_data["threshold"]),
        "positive_class": str(model_data["positive_class"]),
    }


@lru_cache(maxsize=1)
def _load_model() -> dict[str, Any]:
    return _normalize_model(VEGETATION_DAMAGE_MODEL_DATA)


@lru_cache(maxsize=1)
def _load_maturity_model() -> dict[str, Any]:
    return _normalize_model(VEGETATION_MATURITY_MODEL_DATA)


def extract_vegetation_model_features(
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> dict[str, float]:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape HxWx3.")

    rgb = _resize_for_features(rgb_image, max_dimension=max_dimension).astype(np.float32) / 255.0
    resized_mask: np.ndarray | None = None
    if mask is not None:
        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        valid_mask = resized_mask > 0
        if not np.any(valid_mask):
            raise ValueError("Mask does not contain any valid pixels.")
    else:
        valid_mask = np.ones(rgb.shape[:2], dtype=bool)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    tgi_map = -0.5 * ((190.0 * (r - g)) - (120.0 * (r - b)))
    green_mask = (g > r * 1.05) & (g > b * 1.05) & valid_mask

    return {
        "tgi": float(np.mean(tgi_map[valid_mask])),
        "std_g": float(np.std(g[valid_mask])),
        "stand_uniformity_score": _compute_patch_uniformity(green_mask),
    }


def predict_vegetation_damage_from_rgb(
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> VegetationDamagePrediction:
    model = _load_model()
    feature_values = extract_vegetation_model_features(rgb_image, mask=mask, max_dimension=max_dimension)
    features = np.array([feature_values[name] for name in model["feature_names"]], dtype=np.float64)
    scaled = (features - model["mean"]) / model["std"]
    design = np.concatenate([np.ones(1, dtype=np.float64), scaled])
    positive_probability = float(_sigmoid(design @ model["weights"]))
    healthy_probability = float(1.0 - positive_probability)
    predicted_label = model["positive_class"] if positive_probability >= model["threshold"] else "healthy"
    predicted_probability = positive_probability if predicted_label == model["positive_class"] else healthy_probability

    return VegetationDamagePrediction(
        predicted_label=predicted_label,
        probability=predicted_probability,
        confidence=predicted_probability,
        healthy_probability=healthy_probability,
        unhealthy_damaged_probability=positive_probability,
        threshold=model["threshold"],
        feature_values={name: round(value, 6) for name, value in feature_values.items()},
        feature_names=list(model["feature_names"]),
    )


def predict_vegetation_damage_from_path(image_path: str | Path) -> VegetationDamagePrediction:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return predict_vegetation_damage_from_rgb(rgb_image)


def predict_maturity_from_rgb(
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> VegetationMaturityPrediction:
    model = _load_maturity_model()
    feature_values = extract_vegetation_model_features(rgb_image, mask=mask, max_dimension=max_dimension)
    features = np.array([feature_values[name] for name in model["feature_names"]], dtype=np.float64)
    scaled = (features - model["mean"]) / model["std"]
    design = np.concatenate([np.ones(1, dtype=np.float64), scaled])
    positive_probability = float(_sigmoid(design @ model["weights"]))
    predicted_label = model["positive_class"] if positive_probability >= model["threshold"] else "not_mature"
    predicted_probability = positive_probability if predicted_label == model["positive_class"] else (1.0 - positive_probability)

    return VegetationMaturityPrediction(
        predicted_label=predicted_label,
        probability=predicted_probability,
        confidence=predicted_probability,
        maturity_probability=positive_probability,
        threshold=model["threshold"],
        feature_values={name: round(value, 6) for name, value in feature_values.items()},
        feature_names=list(model["feature_names"]),
    )


def predict_maturity_from_path(image_path: str | Path) -> VegetationMaturityPrediction:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return predict_maturity_from_rgb(rgb_image)
