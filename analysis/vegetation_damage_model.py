from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

EPSILON = 1e-6
DEFAULT_FEATURE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

DEFAULT_HEALTH_MODEL_PATH = MODELS_DIR / "health-status" / "rrl_health_mobilenet_rebuilt_best.pt"
DEFAULT_HEALTH_REPORT_PATH = MODELS_DIR / "health-status" / "rrl_health_mobilenet_rebuilt_report.json"
DEFAULT_STAGE_MODEL_PATH = MODELS_DIR / "growth-stage" / "rrl_stage_mobilenet_realistic_best.pt"
DEFAULT_STAGE_REPORT_PATH = MODELS_DIR / "growth-stage" / "Session012_Result_Summary.json"

_MODEL_SETTINGS: dict[str, Any] = {
    "health_model_path": os.getenv("AGRIVISION_HEALTH_MODEL_PATH", str(DEFAULT_HEALTH_MODEL_PATH)),
    "health_report_path": os.getenv("AGRIVISION_HEALTH_REPORT_PATH", str(DEFAULT_HEALTH_REPORT_PATH)),
    "stage_model_path": os.getenv("AGRIVISION_STAGE_MODEL_PATH", str(DEFAULT_STAGE_MODEL_PATH)),
    "stage_report_path": os.getenv("AGRIVISION_STAGE_REPORT_PATH", str(DEFAULT_STAGE_REPORT_PATH)),
    "device": os.getenv("AGRIVISION_MODEL_DEVICE", "").strip(),
}


@dataclass(frozen=True)
class VegetationDamagePrediction:
    predicted_label: str
    probability: float
    confidence: float
    class_probabilities: dict[str, float]
    healthy_probability: float
    unhealthy_damaged_probability: float
    threshold: float
    feature_values: dict[str, float]
    feature_names: list[str]
    class_names: list[str]
    model_name: str
    model_kind: str
    image_size: int


@dataclass(frozen=True)
class VegetationGrowthStagePrediction:
    predicted_label: str
    probability: float
    confidence: float
    class_probabilities: dict[str, float]
    growth_stage_probability: float
    maturity_probability: float
    threshold: float
    feature_values: dict[str, float]
    feature_names: list[str]
    class_names: list[str]
    model_name: str
    model_kind: str
    image_size: int


VegetationMaturityPrediction = VegetationGrowthStagePrediction


@dataclass(frozen=True)
class _HybridCheckpoint:
    checkpoint_path: Path
    model_name: str
    class_names: tuple[str, ...]
    image_size: int
    model_kind: str
    tabular_features: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    model_state_dict: dict[str, torch.Tensor]


class _HybridMobileNetV3Large(nn.Module):
    def __init__(self, num_classes: int, num_tabular_features: int, dropout: float = 0.28) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.num_tabular_features = num_tabular_features

        image_dim = backbone.classifier[0].in_features
        input_dim = image_dim + num_tabular_features
        if num_tabular_features > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 768),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(768, 192),
                nn.Hardswish(),
                nn.Dropout(dropout / 2.0),
                nn.Linear(192, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(1024, num_classes),
            )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.num_tabular_features > 0:
            x = torch.cat([x, tabular], dim=1)
        return self.classifier(x)


def configure_model_paths(
    *,
    health_model_path: str | Path | None = None,
    health_report_path: str | Path | None = None,
    stage_model_path: str | Path | None = None,
    stage_report_path: str | Path | None = None,
    device: str | None = None,
) -> None:
    if health_model_path is not None:
        _MODEL_SETTINGS["health_model_path"] = str(health_model_path)
    if health_report_path is not None:
        _MODEL_SETTINGS["health_report_path"] = str(health_report_path)
    if stage_model_path is not None:
        _MODEL_SETTINGS["stage_model_path"] = str(stage_model_path)
    if stage_report_path is not None:
        _MODEL_SETTINGS["stage_report_path"] = str(stage_report_path)
    if device is not None:
        _MODEL_SETTINGS["device"] = str(device).strip()

    _resolve_device.cache_clear()
    _load_checkpoint.cache_clear()
    _load_model.cache_clear()


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


@lru_cache(maxsize=1)
def _resolve_device() -> torch.device:
    configured = str(_MODEL_SETTINGS.get("device", "") or "").strip()
    if configured:
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_path(model_key: str) -> Path:
    return _resolve_path(str(_MODEL_SETTINGS[model_key]))


@lru_cache(maxsize=4)
def _load_checkpoint(model_key: str) -> _HybridCheckpoint:
    checkpoint_path = _checkpoint_path(model_key)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError(f"Checkpoint is missing model weights: {checkpoint_path}")

    class_names = tuple(str(item) for item in checkpoint.get("class_names", []))
    tabular_features = tuple(str(item) for item in checkpoint.get("tabular_features", []))
    if not class_names:
        raise RuntimeError(f"Checkpoint is missing class_names metadata: {checkpoint_path}")
    if str(checkpoint.get("model_kind", "")).strip().lower() != "hybrid":
        raise RuntimeError(f"Unsupported model_kind in checkpoint: {checkpoint_path}")

    feature_mean = np.asarray(checkpoint.get("feature_mean", []), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std", []), dtype=np.float32)
    if len(tabular_features) != len(feature_mean) or len(tabular_features) != len(feature_std):
        raise RuntimeError(f"Checkpoint feature metadata is inconsistent: {checkpoint_path}")
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    return _HybridCheckpoint(
        checkpoint_path=checkpoint_path,
        model_name=str(checkpoint.get("model_name") or checkpoint_path.stem),
        class_names=class_names,
        image_size=int(checkpoint.get("image_size", 320)),
        model_kind=str(checkpoint.get("model_kind", "hybrid")),
        tabular_features=tabular_features,
        feature_mean=feature_mean,
        feature_std=feature_std,
        model_state_dict=model_state,
    )


@lru_cache(maxsize=4)
def _load_model(model_key: str) -> tuple[_HybridMobileNetV3Large, _HybridCheckpoint]:
    checkpoint = _load_checkpoint(model_key)
    device = _resolve_device()
    model = _HybridMobileNetV3Large(
        num_classes=len(checkpoint.class_names),
        num_tabular_features=len(checkpoint.tabular_features),
    )
    model.load_state_dict(checkpoint.model_state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


@lru_cache(maxsize=8)
def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _ensure_rgb_uint8_array(rgb_image: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb_image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape HxWx3.")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _prepare_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    if mask_array.ndim != 2:
        raise ValueError("Expected mask to have shape HxW or HxWx1.")

    target_h, target_w = shape
    if mask_array.shape != (target_h, target_w):
        mask_array = cv2.resize(
            mask_array.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    valid_mask = mask_array > 0
    if not np.any(valid_mask):
        raise ValueError("Mask does not contain any valid pixels.")
    return valid_mask


def _crop_to_valid_region(rgb_image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    if mask is None:
        return rgb_image, None
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return rgb_image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _resize_rgb_and_mask(
    rgb_image: np.ndarray,
    mask: np.ndarray | None,
    size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    resized_rgb = cv2.resize(rgb_image, (size, size), interpolation=cv2.INTER_AREA)
    resized_mask: np.ndarray | None = None
    if mask is not None:
        resized_mask = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
        if not np.any(resized_mask):
            raise ValueError("Mask does not contain any valid pixels after resizing.")
    return resized_rgb, resized_mask


def _fill_masked_pixels(rgb_image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return rgb_image
    valid_pixels = rgb_image[mask]
    if valid_pixels.size == 0:
        raise ValueError("Mask does not contain any valid pixels.")
    fill_color = np.round(valid_pixels.mean(axis=0)).astype(np.uint8)
    filled = rgb_image.copy()
    filled[~mask] = fill_color
    return filled


def _scale_signed(value: float, clip_limit: float = 1.0) -> float:
    clipped = float(np.clip(value, -clip_limit, clip_limit))
    return (clipped + clip_limit) / (2.0 * clip_limit)


def _scale_range(value: float, min_value: float, max_value: float) -> float:
    if np.isclose(min_value, max_value):
        return 0.5
    clipped = float(np.clip(value, min_value, max_value))
    return (clipped - min_value) / (max_value - min_value)


def _compute_patch_uniformity(mask: np.ndarray, patch_rows: int = 8, patch_cols: int = 8) -> float:
    grid = cv2.resize(mask.astype(np.float32), (patch_cols, patch_rows), interpolation=cv2.INTER_AREA)
    mean_value = float(np.mean(grid))
    std_value = float(np.std(grid))
    if mean_value <= EPSILON:
        return 0.0
    coefficient_of_variation = std_value / (mean_value + EPSILON)
    return float(np.clip(1.0 - coefficient_of_variation, 0.0, 1.0))


def extract_vegetation_model_features(
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
    feature_size: int = DEFAULT_FEATURE_SIZE,
) -> dict[str, float]:
    rgb_uint8 = _ensure_rgb_uint8_array(rgb_image)
    valid_mask = _prepare_mask(mask, rgb_uint8.shape[:2])
    rgb_uint8, valid_mask = _crop_to_valid_region(rgb_uint8, valid_mask)
    if feature_size > 0:
        rgb_uint8, valid_mask = _resize_rgb_and_mask(rgb_uint8, valid_mask, feature_size)

    bgr_array = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    bgr_float = bgr_array.astype(np.float32) / 255.0
    b, g, r = cv2.split(bgr_float)
    valid_pixels = valid_mask if valid_mask is not None else np.ones(rgb_uint8.shape[:2], dtype=bool)
    if not np.any(valid_pixels):
        raise ValueError("Image does not contain any valid pixels for feature extraction.")

    mean_r = float(np.mean(r[valid_pixels]))
    mean_g = float(np.mean(g[valid_pixels]))
    mean_b = float(np.mean(b[valid_pixels]))
    std_r = float(np.std(r[valid_pixels]))
    std_g = float(np.std(g[valid_pixels]))
    std_b = float(np.std(b[valid_pixels]))

    vari_map = (g - r) / (g + r - b + EPSILON)
    gli_map = (2.0 * g - r - b) / (2.0 * g + r + b + EPSILON)
    ngrdi_map = (g - r) / (g + r + EPSILON)
    exg_map = 2.0 * g - r - b
    tgi_map = -0.5 * ((190.0 * (r - g)) - (120.0 * (r - b)))
    mgrvi_map = (g * g - r * r) / (g * g + r * r + EPSILON)

    vari = float(np.mean(vari_map[valid_pixels]))
    gli = float(np.mean(gli_map[valid_pixels]))
    ngrdi = float(np.mean(ngrdi_map[valid_pixels]))
    exg = float(np.mean(exg_map[valid_pixels]))
    tgi = float(np.mean(tgi_map[valid_pixels]))
    mgrvi = float(np.mean(mgrvi_map[valid_pixels]))

    green_mask = (g > r * 1.05) & (g > b * 1.05) & valid_pixels
    green_coverage = float(np.mean(green_mask[valid_pixels]))

    gli_norm = _scale_signed(gli, clip_limit=1.0)
    vari_norm = _scale_signed(vari, clip_limit=1.0)
    ngrdi_norm = _scale_signed(ngrdi, clip_limit=1.0)
    mgrvi_norm = _scale_signed(mgrvi, clip_limit=1.0)
    exg_norm = _scale_range(exg, -1.0, 1.0)
    tgi_norm = _scale_range(tgi, -30.0, 30.0)

    canopy_cover_pct = float(np.clip((0.7 * green_coverage) + (0.3 * gli_norm), 0.0, 1.0))
    lai_proxy = float(np.clip((0.6 * canopy_cover_pct) + (0.4 * vari_norm), 0.0, 1.0))
    spectral_score = float(
        np.clip(
            (0.25 * vari_norm)
            + (0.25 * exg_norm)
            + (0.15 * gli_norm)
            + (0.15 * ngrdi_norm)
            + (0.10 * tgi_norm)
            + (0.10 * mgrvi_norm),
            0.0,
            1.0,
        )
    )
    relative_biomass_score = float(np.clip((0.5 * spectral_score) + (0.5 * canopy_cover_pct), 0.0, 1.0))
    stand_uniformity_score = _compute_patch_uniformity(green_mask.astype(np.float32))
    relative_yield_potential = float(
        np.clip((0.6 * relative_biomass_score) + (0.4 * stand_uniformity_score), 0.0, 1.0)
    )
    baseline_vegetation_score = float(np.clip((0.8 * spectral_score) + (0.2 * canopy_cover_pct), 0.0, 1.0))
    crop_presence_score = float(
        np.clip(
            (0.55 * green_coverage)
            + (0.25 * stand_uniformity_score)
            + (0.20 * relative_biomass_score),
            0.0,
            1.0,
        )
    )
    pseudo_condition_score = float(
        np.clip(
            (0.35 * baseline_vegetation_score)
            + (0.25 * green_coverage)
            + (0.20 * relative_biomass_score)
            + (0.20 * stand_uniformity_score),
            0.0,
            1.0,
        )
    )

    rgb_float = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    dgci_map = (((hue - 60.0) / 60.0) + (1.0 - saturation) + (1.0 - brightness)) / 3.0
    dgci = float(np.mean(np.clip(dgci_map, 0.0, 1.0)[valid_pixels]))

    b_255 = bgr_array[:, :, 0].astype(np.float32)
    g_255 = bgr_array[:, :, 1].astype(np.float32)
    r_255 = bgr_array[:, :, 2].astype(np.float32)
    cive_map = (0.441 * r_255) - (0.811 * g_255) + (0.385 * b_255) + 18.787
    cive = float(np.mean(cive_map[valid_pixels]))

    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "vari": vari,
        "gli": gli,
        "ngrdi": ngrdi,
        "exg": exg,
        "tgi": tgi,
        "mgrvi": mgrvi,
        "green_coverage": green_coverage,
        "canopy_cover_pct": canopy_cover_pct,
        "lai_proxy": lai_proxy,
        "relative_biomass_score": relative_biomass_score,
        "stand_uniformity_score": stand_uniformity_score,
        "relative_yield_potential": relative_yield_potential,
        "spectral_score": spectral_score,
        "baseline_vegetation_score": baseline_vegetation_score,
        "crop_presence_score": crop_presence_score,
        "pseudo_condition_score": pseudo_condition_score,
        "dgci": dgci,
        "cive": cive,
    }


def _image_tensor_from_rgb(
    rgb_image: np.ndarray,
    mask: np.ndarray | None,
    image_size: int,
) -> torch.Tensor:
    prepared_rgb = _fill_masked_pixels(rgb_image, mask)
    pil_image = Image.fromarray(prepared_rgb, mode="RGB")
    return _eval_transform(image_size)(pil_image).unsqueeze(0)


def _predict_with_checkpoint(
    model_key: str,
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, _HybridCheckpoint, dict[str, float]]:
    model, checkpoint = _load_model(model_key)
    device = _resolve_device()

    rgb_uint8 = _ensure_rgb_uint8_array(rgb_image)
    valid_mask = _prepare_mask(mask, rgb_uint8.shape[:2])
    cropped_rgb, cropped_mask = _crop_to_valid_region(rgb_uint8, valid_mask)

    image_tensor = _image_tensor_from_rgb(cropped_rgb, cropped_mask, checkpoint.image_size).to(device)
    all_features = extract_vegetation_model_features(
        cropped_rgb,
        mask=cropped_mask.astype(np.uint8) if cropped_mask is not None else None,
    )
    selected_features = {
        name: round(float(all_features[name]), 6)
        for name in checkpoint.tabular_features
    }
    feature_vector = np.asarray(
        [float(all_features[name]) for name in checkpoint.tabular_features],
        dtype=np.float32,
    )
    feature_vector = (feature_vector - checkpoint.feature_mean) / checkpoint.feature_std
    tabular_tensor = torch.from_numpy(feature_vector).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor, tabular_tensor)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    return probabilities, checkpoint, selected_features


def predict_vegetation_damage_from_rgb(
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> VegetationDamagePrediction:
    probabilities, checkpoint, feature_values = _predict_with_checkpoint(
        "health_model_path",
        rgb_image,
        mask=mask,
    )
    class_probabilities = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(checkpoint.class_names)
    }
    predicted_index = int(np.argmax(probabilities))
    predicted_label = checkpoint.class_names[predicted_index]
    predicted_probability = float(probabilities[predicted_index])

    return VegetationDamagePrediction(
        predicted_label=predicted_label,
        probability=predicted_probability,
        confidence=predicted_probability,
        class_probabilities=class_probabilities,
        healthy_probability=float(class_probabilities.get("healthy", 0.0)),
        unhealthy_damaged_probability=float(class_probabilities.get("unhealthy_damaged", 0.0)),
        threshold=0.5,
        feature_values=feature_values,
        feature_names=list(checkpoint.tabular_features),
        class_names=list(checkpoint.class_names),
        model_name=checkpoint.model_name,
        model_kind=checkpoint.model_kind,
        image_size=checkpoint.image_size,
    )


def predict_vegetation_damage_from_path(image_path: str | Path) -> VegetationDamagePrediction:
    try:
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb_array = np.asarray(rgb, dtype=np.uint8)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc
    return predict_vegetation_damage_from_rgb(rgb_array)


def predict_growth_stage_from_rgb(
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> VegetationGrowthStagePrediction:
    probabilities, checkpoint, feature_values = _predict_with_checkpoint(
        "stage_model_path",
        rgb_image,
        mask=mask,
    )
    class_probabilities = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(checkpoint.class_names)
    }
    predicted_index = int(np.argmax(probabilities))
    predicted_label = checkpoint.class_names[predicted_index]
    predicted_probability = float(probabilities[predicted_index])

    mature_probability = 0.0
    for class_name, probability in class_probabilities.items():
        if class_name.strip().lower() == "mature (senescence)":
            mature_probability = float(probability)
            break

    return VegetationGrowthStagePrediction(
        predicted_label=predicted_label,
        probability=predicted_probability,
        confidence=predicted_probability,
        class_probabilities=class_probabilities,
        growth_stage_probability=predicted_probability,
        maturity_probability=mature_probability,
        threshold=0.5,
        feature_values=feature_values,
        feature_names=list(checkpoint.tabular_features),
        class_names=list(checkpoint.class_names),
        model_name=checkpoint.model_name,
        model_kind=checkpoint.model_kind,
        image_size=checkpoint.image_size,
    )


def predict_growth_stage_from_path(image_path: str | Path) -> VegetationGrowthStagePrediction:
    try:
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb_array = np.asarray(rgb, dtype=np.uint8)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc
    return predict_growth_stage_from_rgb(rgb_array)


def predict_maturity_from_rgb(
    rgb_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> VegetationMaturityPrediction:
    return predict_growth_stage_from_rgb(rgb_image, mask=mask)


def predict_maturity_from_path(image_path: str | Path) -> VegetationMaturityPrediction:
    return predict_growth_stage_from_path(image_path)
