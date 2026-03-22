import unittest

import numpy as np
from PIL import Image

from analysis.vegetation_damage_model import (
    predict_growth_stage_from_rgb,
    predict_vegetation_damage_from_rgb,
)
from app import (
    _build_effective_crop_mask,
    _masked_rgb_metrics,
    _trained_damage_model_result,
    _weighted_health_score,
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

    def test_textured_senescent_crop_mask_is_not_removed_as_deadspace(self) -> None:
        image = self._make_textured_senescent_field()
        full_mask = Image.new("L", image.size, 255)

        effective_mask = _build_effective_crop_mask(image, full_mask)
        coverage = np.asarray(effective_mask, dtype=np.uint8).mean() / 255.0

        self.assertGreater(coverage, 0.85)

    def test_masked_metrics_keep_textured_senescent_pixels_when_mask_is_explicit(self) -> None:
        image = self._make_textured_senescent_field()
        full_mask = Image.new("L", image.size, 255)

        metrics = _masked_rgb_metrics(image, full_mask, include_soft_deadspace=False)

        self.assertGreater(metrics["valid_count"], 0)
        self.assertGreater(metrics["valid_pixel_ratio"], 0.95)


if __name__ == "__main__":
    unittest.main()
