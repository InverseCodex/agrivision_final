import unittest

import numpy as np

from analysis.vegetation_damage_model import (
    predict_maturity_from_rgb,
    predict_vegetation_damage_from_rgb,
)
from app import _trained_damage_model_result


class VegetationDamageModelTests(unittest.TestCase):
    def test_green_image_predicts_healthy(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[..., 0] = 40
        image[..., 1] = 180
        image[..., 2] = 35

        prediction = predict_vegetation_damage_from_rgb(image)

        self.assertEqual(prediction.predicted_label, "healthy")
        self.assertGreater(prediction.confidence, 0.9)
        self.assertEqual(prediction.feature_names, ["tgi", "std_g", "stand_uniformity_score"])

    def test_brown_image_predicts_unhealthy_damaged(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[..., 0] = 150
        image[..., 1] = 90
        image[..., 2] = 80

        prediction = predict_vegetation_damage_from_rgb(image)

        self.assertEqual(prediction.predicted_label, "unhealthy_damaged")
        self.assertGreater(prediction.confidence, 0.9)
        self.assertIn("stand_uniformity_score", prediction.feature_values)

    def test_maturity_model_overrides_health_band_to_mature(self) -> None:
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[..., 0] = 220
        image[..., 1] = 240
        image[..., 2] = 60

        health_prediction = predict_vegetation_damage_from_rgb(image)
        maturity_prediction = predict_maturity_from_rgb(image)
        result = _trained_damage_model_result(
            health_prediction,
            health_score=83.5,
            maturity_prediction=maturity_prediction,
        )

        self.assertEqual(maturity_prediction.predicted_label, "healthy_mature")
        self.assertEqual(result["health_band"], "mature")
        self.assertEqual(result["maturity_label"], "mature")
        self.assertGreater(result["maturity_probability"], 0.76)
        self.assertEqual(result["health_score"], 83.5)


if __name__ == "__main__":
    unittest.main()
