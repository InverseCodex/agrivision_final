import base64
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from analysis.vegetation_damage_model import (
    predict_growth_stage_from_rgb,
    predict_vegetation_damage_from_rgb,
)
from app import (
    _default_full_selection_points,
    _extract_field_stage_model,
    _is_full_frame_quad,
    _merge_field_stage_into_model_result,
    _masked_rgb_metrics,
    _phenological_stage_label,
    _trained_damage_model_result,
    _weighted_health_score,
    analyze_freeform_cropped_segments,
)


class VegetationDamageModelTests(unittest.TestCase):
    def _make_textured_senescent_field(self) -> Image.Image:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if (x // 6) % 2 == 0:
                    image[y, x] = (176, 186, 118)
                else:
                    image[y, x] = (128, 150, 74)
                if (y // 12) % 2 == 1:
                    image[y, x] = np.clip(image[y, x] - np.array((18, 10, 8)), 0, 255)
        return Image.fromarray(image, mode="RGB")

    def test_green_image_predicts_healthy(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[..., 0] = 40
        image[..., 1] = 180
        image[..., 2] = 35

        prediction = predict_vegetation_damage_from_rgb(image)

        self.assertEqual(prediction.predicted_label, "healthy")
        self.assertGreater(prediction.confidence, 0.9)
        self.assertEqual(prediction.feature_names, ["tgi", "dgci", "cive", "mgrvi"])

    def test_brown_image_predicts_unhealthy_damaged(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[..., 0] = 150
        image[..., 1] = 90
        image[..., 2] = 80

        prediction = predict_vegetation_damage_from_rgb(image)

        self.assertEqual(prediction.predicted_label, "unhealthy_damaged")
        self.assertGreater(prediction.confidence, 0.9)
        self.assertIn("mgrvi", prediction.feature_values)
        self.assertEqual(set(prediction.class_probabilities.keys()), {"healthy", "unhealthy_damaged"})

    def test_growth_stage_model_returns_stage_probabilities(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[..., 0] = 40
        image[..., 1] = 180
        image[..., 2] = 35

        stage_prediction = predict_growth_stage_from_rgb(image)

        self.assertEqual(stage_prediction.feature_names, ["exg", "ngrdi", "vari", "tgi", "gli", "mgrvi"])
        self.assertIn(stage_prediction.predicted_label, stage_prediction.class_names)
        self.assertAlmostEqual(sum(stage_prediction.class_probabilities.values()), 1.0, places=4)

    def test_growth_stage_result_can_override_health_band_to_mature(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[..., 0] = 40
        image[..., 1] = 180
        image[..., 2] = 35

        health_prediction = predict_vegetation_damage_from_rgb(image)
        stage_prediction = type(
            "StagePrediction",
            (),
            {
                "predicted_label": "Mature (Senescence)",
                "growth_stage_probability": 0.88,
                "probability": 0.88,
                "confidence": 0.88,
                "maturity_probability": 0.88,
                "class_probabilities": {
                    "Early Vegetative": 0.04,
                    "Late Vegetative": 0.03,
                    "Tasseling": 0.05,
                    "Mature (Senescence)": 0.88,
                },
                "feature_values": {"exg": 0.1, "ngrdi": 0.2, "vari": 0.3, "tgi": 5.0, "gli": 0.4, "mgrvi": 0.5},
                "feature_names": ["exg", "ngrdi", "vari", "tgi", "gli", "mgrvi"],
                "model_name": "stage-test",
                "model_kind": "hybrid",
            },
        )()
        result = _trained_damage_model_result(
            health_prediction,
            health_score=83.5,
            stage_prediction=stage_prediction,
        )

        self.assertEqual(result["health_band"], "mature")
        self.assertEqual(result["maturity_label"], "mature")
        self.assertEqual(result["growth_stage_label"], "Mature (Senescence)")
        self.assertGreater(result["maturity_probability"], 0.8)
        self.assertEqual(result["health_score"], 83.5)

    def test_health_score_improves_with_stronger_metrics(self) -> None:
        score_low = _weighted_health_score(
            vigor_score=20,
            biomass_score=20,
            canopy_score=20,
            uniformity_score=20,
        )
        score_high = _weighted_health_score(
            vigor_score=80,
            biomass_score=80,
            canopy_score=80,
            uniformity_score=80,
        )

        self.assertGreater(score_high, score_low)

    def test_masked_metrics_keep_textured_senescent_pixels_when_mask_is_explicit(self) -> None:
        image = self._make_textured_senescent_field()
        full_mask = Image.new("L", image.size, 255)

        metrics = _masked_rgb_metrics(image, full_mask)

        self.assertGreater(metrics["valid_count"], 0)
        self.assertGreater(metrics["valid_pixel_ratio"], 0.95)

    def test_field_stage_can_be_merged_without_overriding_segment_health_band(self) -> None:
        analysis = {
            "report": {
                "model_result": {
                    "growth_stage_label": "Tasseling",
                    "growth_stage_probability": 0.91,
                    "maturity_probability": 0.12,
                    "growth_stage_feature_values": {"gli": 0.44, "vari": 0.31},
                    "growth_stage_feature_names": ["gli", "vari"],
                }
            }
        }

        field_stage = _extract_field_stage_model(analysis)
        merged = _merge_field_stage_into_model_result(
            {
                "health_band": "unhealthy_damaged",
                "health_score": 41.2,
                "confidence": 0.77,
            },
            field_stage,
        )

        self.assertEqual(field_stage["display_label"], "Tasseling")
        self.assertEqual(merged["health_band"], "unhealthy_damaged")
        self.assertEqual(merged["growth_stage_label"], "Tasseling")
        self.assertAlmostEqual(merged["growth_stage_probability"], 0.91, places=4)
        self.assertEqual(merged["growth_stage_feature_names"], ["gli", "vari"])

    def test_phenological_stage_label_does_not_invent_binary_stage_text(self) -> None:
        self.assertEqual(
            _phenological_stage_label(
                {
                    "growth_stage_label": "",
                    "maturity_label": "not_mature",
                    "maturity_probability": 0.38,
                    "health_band": "healthy",
                }
            ),
            "Stage not available",
        )

    def test_default_full_selection_points_cover_entire_image(self) -> None:
        self.assertEqual(
            _default_full_selection_points(),
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

    def test_full_frame_quad_detection_matches_image_corners(self) -> None:
        self.assertTrue(_is_full_frame_quad([(0, 0), (159, 0), (159, 95), (0, 95)], 160, 96))
        self.assertFalse(_is_full_frame_quad([(12, 6), (150, 6), (150, 90), (12, 90)], 160, 96))

    def test_segment_analysis_preserves_selection_geometry_without_rectification(self) -> None:
        width, height = 128, 96
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = 58
        image[..., 1] = 146
        image[..., 2] = 74
        source_image = Image.fromarray(image, mode="RGB")
        points = [(0.25, 0.05), (0.75, 0.1), (0.95, 0.95), (0.05, 0.9)]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "field.png"
            source_image.save(image_path)

            result = analyze_freeform_cropped_segments(
                image_path,
                "geometry-test",
                points,
                4,
                4,
                field_stage_model={},
            )

        mask = Image.new("L", (width, height), 0)
        px_points = [
            (
                min(max(int(round(x_norm * (width - 1))), 0), width - 1),
                min(max(int(round(y_norm * (height - 1))), 0), height - 1),
            )
            for x_norm, y_norm in points
        ]
        ImageDraw.Draw(mask).polygon(px_points, fill=255)
        bbox = mask.getbbox()
        self.assertIsNotNone(bbox)
        expected_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

        _, encoded = result["cropped_original_data_url"].split(",", 1)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as rendered:
            self.assertEqual(rendered.size, expected_size)


if __name__ == "__main__":
    unittest.main()
