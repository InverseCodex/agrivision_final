from dataclasses import dataclass
from typing import List


@dataclass
class IndexSnapshot:
    ndvi: float
    evi: float
    savi: float
    gndvi: float
    ndre: float


@dataclass
class RGBSnapshot:
    # Mean channel values from RGB image (0..255)
    mean_red: float
    mean_green: float
    mean_blue: float
    # Optional canopy estimate from segmentation (0..1). Use -1 if unknown.
    green_coverage: float = -1.0
    # Optional dry-canopy estimate (0..1) for mature crops that are no longer vividly green.
    dry_coverage: float = -1.0
    # Source camera profile used for RGB heuristics.
    camera_model: str = "generic-rgb"


@dataclass
class FieldContext:
    crop_name: str = "Crop"
    growth_stage: str = "mid-season"
    rainfall_last_7d_mm: float = 0.0
    avg_temp_c: float = 0.0


@dataclass
class JudgementResult:
    health_score: float
    health_band: str
    confidence: float
    main_findings: List[str]


@dataclass
class FarmerReport:
    one_line_summary: str
    simple_explanation: str
    recommendations: List[str]
    model_result: JudgementResult


class VegetationJudgeModel:
    """
    Lightweight explainable scorer that behaves like an AI judge:
    it combines multiple vegetation indices and environmental context
    to estimate crop health and produce interpretable findings.
    """

    def __init__(self) -> None:
        self.weights = {
            "ndvi": 0.30,
            "evi": 0.25,
            "savi": 0.20,
            "gndvi": 0.15,
            "ndre": 0.10,
        }

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _normalize_index(value: float) -> float:
        # Maps [-1, 1] to [0, 1]
        return (value + 1.0) / 2.0

    @staticmethod
    def _band(score: float) -> str:
        if score >= 0.75:
            return "healthy"
        if score >= 0.55:
            return "watch"
        if score >= 0.35:
            return "stressed"
        return "critical"

    @staticmethod
    def _is_mature_stage(growth_stage: str) -> bool:
        stage = growth_stage.strip().lower()
        return any(token in stage for token in ("mature", "maturity", "ripening", "senescence", "senescent", "harvest"))

    def predict(self, snapshot: IndexSnapshot, context: FieldContext) -> JudgementResult:
        normalized = {
            "ndvi": self._normalize_index(snapshot.ndvi),
            "evi": self._normalize_index(snapshot.evi),
            "savi": self._normalize_index(snapshot.savi),
            "gndvi": self._normalize_index(snapshot.gndvi),
            "ndre": self._normalize_index(snapshot.ndre),
        }

        weighted_score = sum(normalized[k] * self.weights[k] for k in self.weights)

        # Small context adjustments
        if context.rainfall_last_7d_mm < 10:
            weighted_score -= 0.03
        if context.avg_temp_c > 35:
            weighted_score -= 0.03
        if context.avg_temp_c < 10:
            weighted_score -= 0.02

        health_score = self._clamp(weighted_score, 0.0, 1.0)
        health_band = self._band(health_score)

        values = [snapshot.ndvi, snapshot.evi, snapshot.savi, snapshot.gndvi, snapshot.ndre]
        spread = max(values) - min(values)
        confidence = self._clamp(1.0 - (spread / 2.0), 0.40, 0.98)

        findings: List[str] = []
        if snapshot.ndvi < 0.35:
            findings.append("Plant vigor is low in many areas.")
        if snapshot.evi < 0.30:
            findings.append("Canopy growth looks limited and may need support.")
        if snapshot.gndvi < 0.30:
            findings.append("Possible nitrogen stress is visible.")
        if snapshot.savi < 0.30:
            findings.append("Soil influence is high; plant cover may be sparse.")
        if snapshot.ndre < 0.28:
            findings.append("Early stress signs are present in leaves.")
        if not findings:
            findings.append("Index patterns indicate generally stable crop health.")

        return JudgementResult(
            health_score=round(health_score, 3),
            health_band=health_band,
            confidence=round(confidence, 3),
            main_findings=findings,
        )

    def predict_from_rgb_proxies(
        self,
        vari: float,
        gli: float,
        ngrdi: float,
        exg: float,
        context: FieldContext,
        green_coverage: float = -1.0,
        dry_coverage: float = -1.0,
        camera_model: str = "generic-rgb",
    ) -> JudgementResult:
        # RGB proxy model for non-multispectral cameras.
        camera_key = camera_model.strip().lower()
        is_dji_mini_4_pro = "mini 4 pro" in camera_key
        proxy_weights = {
            "vari": 0.30,
            "gli": 0.25,
            "ngrdi": 0.20,
            "exg": 0.25,
        }
        if is_dji_mini_4_pro:
            # DJI Mini 4 Pro RGB tends to preserve more yellow/brown canopy detail than
            # simpler "is it green" heuristics expect, so shift weight toward canopy structure.
            proxy_weights = {
                "vari": 0.24,
                "gli": 0.31,
                "ngrdi": 0.17,
                "exg": 0.28,
            }

        normalized = {
            "vari": self._normalize_index(vari),
            "gli": self._normalize_index(gli),
            "ngrdi": self._normalize_index(ngrdi),
            # ExG typically in a broader range; squeeze to [-1,1] first.
            "exg": self._normalize_index(self._clamp(exg / 2.0, -1.0, 1.0)),
        }

        weighted_score = sum(normalized[k] * proxy_weights[k] for k in proxy_weights)

        if 0.0 <= green_coverage <= 1.0:
            weighted_score = (weighted_score * 0.8) + (green_coverage * 0.2)

        if 0.0 <= dry_coverage <= 1.0 and 0.0 <= green_coverage <= 1.0:
            mature_canopy_signal = (dry_coverage >= 0.35 and green_coverage <= 0.35)
            if mature_canopy_signal:
                # Guard against false stress on mature / senescent crops captured by RGB only.
                weighted_score += 0.08 if is_dji_mini_4_pro else 0.05
        else:
            mature_canopy_signal = False

        # Context adjustments
        if context.rainfall_last_7d_mm < 10:
            weighted_score -= 0.03
        if context.avg_temp_c > 35:
            weighted_score -= 0.03
        if context.avg_temp_c < 10:
            weighted_score -= 0.02

        health_score = self._clamp(weighted_score, 0.0, 1.0)
        health_band = self._band(health_score)
        mature_stage = self._is_mature_stage(context.growth_stage)
        mature_override = mature_canopy_signal and (mature_stage or health_band in {"watch", "stressed"})
        if mature_override and health_band != "critical":
            health_band = "mature"

        values = [vari, gli, ngrdi, self._clamp(exg / 2.0, -1.0, 1.0)]
        spread = max(values) - min(values)
        # RGB-only model has lower theoretical certainty than multispectral.
        confidence = self._clamp(0.82 - (spread * 0.12), 0.45, 0.86)
        if 0.0 <= green_coverage <= 1.0:
            confidence = self._clamp(confidence + 0.04, 0.45, 0.90)

        findings: List[str] = []
        weak_green_threshold = 0.02 if is_dji_mini_4_pro else 0.05
        thin_canopy_threshold = 0.03 if is_dji_mini_4_pro else 0.05
        weak_signal_threshold = 0.02 if is_dji_mini_4_pro else 0.05
        sparse_cover_threshold = 0.30 if is_dji_mini_4_pro else 0.45

        if mature_override:
            findings.append("Canopy color is consistent with maturity or harvest stage more than acute stress.")
        if vari < weak_green_threshold or ngrdi < weak_green_threshold:
            findings.append("Green vigor appears weak in parts of the field.")
        if gli < thin_canopy_threshold:
            findings.append("Leaf density looks thin; canopy may be uneven.")
        if exg < weak_signal_threshold:
            findings.append("Vegetation signal is low versus soil/background.")
        if 0.0 <= green_coverage <= 1.0 and green_coverage < sparse_cover_threshold:
            findings.append("Estimated green cover is below ideal range.")
        if 0.0 <= dry_coverage <= 1.0 and dry_coverage >= 0.35 and green_coverage < 0.35:
            findings.append("Dry canopy is prominent; this may indicate crop maturity, not only active stress.")
        if not findings:
            if health_band in {"stressed", "critical"}:
                findings.append("RGB proxy score indicates stress, even if visual color contrast seems moderate.")
            elif health_band == "mature":
                findings.append("RGB proxy score suggests a mature field with reduced green color but preserved canopy.")
            else:
                findings.append("RGB vegetation proxies indicate fairly stable field condition.")

        return JudgementResult(
            health_score=round(health_score, 3),
            health_band=health_band,
            confidence=round(confidence, 3),
            main_findings=findings,
        )
