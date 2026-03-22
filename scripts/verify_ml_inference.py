from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.vegetation_damage_model import (
    predict_growth_stage_from_rgb,
    predict_vegetation_damage_from_rgb,
)
from app import analyze_freeform_cropped_segments, run_rgb_analysis


def _solid_rgb(red: int, green: int, blue: int, size: int = 160) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = red
    image[..., 1] = green
    image[..., 2] = blue
    return image


def _assert_probability_distribution(probabilities: dict[str, float], expected_count: int) -> None:
    if len(probabilities) != expected_count:
        raise AssertionError(f"Expected {expected_count} classes, got {len(probabilities)}.")
    total = sum(float(value) for value in probabilities.values())
    if not np.isclose(total, 1.0, atol=1e-4):
        raise AssertionError(f"Probabilities must sum to 1.0, got {total}.")


def main() -> None:
    green_image = _solid_rgb(40, 180, 35)
    brown_image = _solid_rgb(150, 90, 80)

    green_health = predict_vegetation_damage_from_rgb(green_image)
    brown_health = predict_vegetation_damage_from_rgb(brown_image)
    green_stage = predict_growth_stage_from_rgb(green_image)

    if green_health.predicted_label != "healthy":
        raise AssertionError(f"Expected green sample to be healthy, got {green_health.predicted_label}.")
    if brown_health.predicted_label != "unhealthy_damaged":
        raise AssertionError(f"Expected brown sample to be unhealthy_damaged, got {brown_health.predicted_label}.")
    if green_health.feature_names != ["tgi", "dgci", "cive", "mgrvi"]:
        raise AssertionError(f"Unexpected health feature names: {green_health.feature_names}")
    if green_stage.feature_names != ["exg", "ngrdi", "vari", "tgi", "gli", "mgrvi"]:
        raise AssertionError(f"Unexpected stage feature names: {green_stage.feature_names}")

    _assert_probability_distribution(green_health.class_probabilities, expected_count=2)
    _assert_probability_distribution(green_stage.class_probabilities, expected_count=4)

    with tempfile.TemporaryDirectory(prefix="agrivision-ml-smoke-") as temp_dir:
        temp_path = Path(temp_dir) / "green_sample.png"
        Image.fromarray(green_image, mode="RGB").save(temp_path)

        analysis_result = run_rgb_analysis(temp_path, user_id="smoke-test", image_id="green-sample")
        model_result = analysis_result["analysis"]["report"]["model_result"]
        if model_result["health_band"] not in {"healthy", "mature"}:
            raise AssertionError(f"Unexpected health band from run_rgb_analysis: {model_result['health_band']}")
        if not model_result.get("growth_stage_label"):
            raise AssertionError("Growth stage label missing from run_rgb_analysis result.")

        crop_result = analyze_freeform_cropped_segments(
            temp_path,
            image_id="green-sample",
            points=[(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)],
            grid_rows=4,
            grid_cols=4,
        )
        crop_model_result = crop_result["crop_analysis"]["report"]["model_result"]
        if not crop_model_result.get("growth_stage_label"):
            raise AssertionError("Growth stage label missing from crop rerun result.")
        if not crop_result.get("segments"):
            raise AssertionError("Crop rerun did not return any segments.")

    print("ML inference smoke verification passed.")


if __name__ == "__main__":
    main()
