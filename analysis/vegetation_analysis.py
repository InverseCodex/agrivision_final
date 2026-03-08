from __future__ import annotations

from typing import Iterable, List, Tuple

from .model import FieldContext, FarmerReport, IndexSnapshot, RGBSnapshot, VegetationJudgeModel


def compute_index(numerator: Iterable[float], denominator: Iterable[float], eps: float = 1e-8) -> List[float]:
    """
    Generic index helper:
    index = numerator / (denominator + eps)
    """
    out: List[float] = []
    for n, d in zip(numerator, denominator):
        out.append(float(n) / (float(d) + eps))
    return out


def _recommendations(snapshot: IndexSnapshot, context: FieldContext, band: str) -> List[str]:
    recs: List[str] = []

    if band in {"stressed", "critical"}:
        recs.append("Check irrigation first, especially in dry or patchy areas.")
        recs.append("Inspect for pests and leaf damage in low-index zones.")
        recs.append("Do spot field visits in the next 24-48 hours.")

    if snapshot.gndvi < 0.30 or snapshot.ndre < 0.28:
        recs.append("Review nitrogen plan and consider split fertilizer application.")

    if snapshot.savi < 0.30:
        recs.append("Improve ground cover to reduce soil exposure and moisture loss.")

    if context.rainfall_last_7d_mm < 10:
        recs.append("Rainfall has been low; monitor soil moisture daily this week.")

    if context.avg_temp_c > 35:
        recs.append("High heat risk: irrigate early morning or late afternoon.")

    if not recs:
        recs.append("Maintain current farm practices and continue weekly monitoring.")
        recs.append("Re-scan after 5-7 days to confirm stability.")

    # Keep output concise for simple farmer-facing delivery.
    return recs[:6]


def _recommendations_rgb(
    vari: float,
    gli: float,
    ngrdi: float,
    exg: float,
    green_coverage: float,
    context: FieldContext,
    band: str,
) -> List[str]:
    recs: List[str] = []

    if band in {"stressed", "critical"}:
        recs.append("Inspect weak patches on-site within 24-48 hours.")
        recs.append("Check irrigation uniformity and soil moisture at root level.")

    if vari < 0.05 or ngrdi < 0.05 or exg < 0.05:
        recs.append("Likely low canopy vigor: review nutrient and watering schedule.")

    if 0.0 <= green_coverage <= 1.0 and green_coverage < 0.45:
        recs.append("Plant cover appears sparse; consider replanting gaps if needed.")

    if context.rainfall_last_7d_mm < 10:
        recs.append("Low recent rain: prioritize moisture checks this week.")

    if context.avg_temp_c > 35:
        recs.append("Heat stress risk: irrigate early morning or late afternoon.")

    recs.append("For best accuracy, fly at similar time/light each scan for comparison.")
    recs.append("RGB drone analysis is a proxy; confirm critical zones with field scouting.")

    return recs[:7]


def rgb_to_vegetation_proxies(snapshot: RGBSnapshot) -> Tuple[float, float, float, float]:
    """
    RGB-only vegetation proxy indices (usable with DJI Mini 4 Pro RGB camera).
    Returns: VARI, GLI, NGRDI, ExG
    """
    r = float(snapshot.mean_red)
    g = float(snapshot.mean_green)
    b = float(snapshot.mean_blue)
    eps = 1e-8

    vari = (g - r) / (g + r - b + eps)
    gli = (2 * g - r - b) / (2 * g + r + b + eps)
    ngrdi = (g - r) / (g + r + eps)
    exg = (2 * g - r - b) / 255.0
    return vari, gli, ngrdi, exg


def interpret_field_from_indices(snapshot: IndexSnapshot, context: FieldContext) -> FarmerReport:
    model = VegetationJudgeModel()
    result = model.predict(snapshot, context)

    if result.health_band == "healthy":
        summary = f"{context.crop_name}: crop condition looks healthy."
    elif result.health_band == "watch":
        summary = f"{context.crop_name}: mostly okay, but some areas need attention."
    elif result.health_band == "stressed":
        summary = f"{context.crop_name}: signs of stress detected, action is needed soon."
    else:
        summary = f"{context.crop_name}: serious stress detected, prioritize field intervention now."

    simple_explanation = (
        f"AI judged field status as '{result.health_band.upper()}' "
        f"with confidence {int(result.confidence * 100)}%. "
        f"The model combined NDVI, EVI, SAVI, GNDVI, and NDRE to produce this result."
    )

    return FarmerReport(
        one_line_summary=summary,
        simple_explanation=simple_explanation,
        recommendations=_recommendations(snapshot, context, result.health_band),
        model_result=result,
    )


def interpret_field_from_rgb(snapshot: RGBSnapshot, context: FieldContext) -> FarmerReport:
    model = VegetationJudgeModel()
    vari, gli, ngrdi, exg = rgb_to_vegetation_proxies(snapshot)

    result = model.predict_from_rgb_proxies(
        vari=vari,
        gli=gli,
        ngrdi=ngrdi,
        exg=exg,
        context=context,
        green_coverage=snapshot.green_coverage,
    )

    if result.health_band == "healthy":
        summary = f"{context.crop_name}: RGB scan suggests healthy field condition."
    elif result.health_band == "watch":
        summary = f"{context.crop_name}: RGB scan shows moderate condition; monitor weak spots."
    elif result.health_band == "stressed":
        summary = f"{context.crop_name}: RGB scan indicates stress signs; intervention is recommended."
    else:
        summary = f"{context.crop_name}: RGB scan indicates severe stress; act immediately."

    simple_explanation = (
        f"AI judged field status as '{result.health_band.upper()}' "
        f"with confidence {int(result.confidence * 100)}%. "
        "Result is based on RGB vegetation proxies (VARI/GLI/NGRDI/ExG), "
        "not true NDVI/NDRE from multispectral sensors."
    )

    return FarmerReport(
        one_line_summary=summary,
        simple_explanation=simple_explanation,
        recommendations=_recommendations_rgb(
            vari=vari,
            gli=gli,
            ngrdi=ngrdi,
            exg=exg,
            green_coverage=snapshot.green_coverage,
            context=context,
            band=result.health_band,
        ),
        model_result=result,
    )
