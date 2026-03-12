#!/usr/bin/env python3
"""
MATLAB-style visualization utility for RGB vegetation analysis outputs.

Usage examples:
  python scripts/plot_analysis_visuals.py --input path/to/result.json --out-dir analysis/plots
  python scripts/plot_analysis_visuals.py --input path/to/crop_response.json --out-dir analysis/plots
  python scripts/plot_analysis_visuals.py --input data/results --out-dir analysis/plots

The script supports:
  - single analysis JSON payloads from the website
  - crop-analysis response JSON payloads with per-segment data
  - directories containing many JSON analysis reports
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

BANDS = ["healthy", "watch", "stressed", "critical", "na"]
BAND_TO_SCORE = {"healthy": 4.0, "watch": 3.0, "stressed": 2.0, "critical": 1.0, "na": 0.0}


@dataclass
class AnalysisRecord:
    image_id: str
    created_at: str
    health_score: float
    confidence: float
    health_band: str
    mean_red: float
    mean_green: float
    mean_blue: float
    green_coverage_pct: float
    stress_zone_pct: float
    vigor_score: float
    vari: float
    gli: float
    ngrdi: float
    exg: float
    tgi: float
    mgrvi: float
    lai_proxy: float
    canopy_cover_pct: float
    biomass_score: float
    uniformity_score: float
    yield_potential_pct: float


def _safe_float(value: Any, fallback: float = float("nan")) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _is_analysis_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("report"), dict) and isinstance(payload.get("rgb_indices"), dict)


def _extract_analysis_payload(payload: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return None, []
    if _is_analysis_payload(payload):
        return payload, []
    crop_analysis = payload.get("crop_analysis")
    segments = payload.get("segments") or []
    if isinstance(crop_analysis, dict) and _is_analysis_payload(crop_analysis):
        valid_segments = [seg for seg in segments if isinstance(seg, dict)]
        return crop_analysis, valid_segments
    return None, []


def _record_from_payload(payload: dict[str, Any], source_name: str) -> AnalysisRecord | None:
    report = payload.get("report") or {}
    model_result = report.get("model_result") or {}
    rgb_indices = payload.get("rgb_indices") or {}
    farmer = payload.get("farmer_features") or {}
    veg = payload.get("vegetation_indices_analysis") or {}
    inputs = payload.get("input") or {}

    band = str(model_result.get("health_band") or "na").strip().lower()
    if band not in BAND_TO_SCORE:
        band = "na"

    return AnalysisRecord(
        image_id=str(payload.get("image_id") or source_name),
        created_at=str(payload.get("created_at") or ""),
        health_score=_safe_float(model_result.get("health_score")),
        confidence=_safe_float(model_result.get("confidence")),
        health_band=band,
        mean_red=_safe_float(inputs.get("mean_red")),
        mean_green=_safe_float(inputs.get("mean_green")),
        mean_blue=_safe_float(inputs.get("mean_blue")),
        green_coverage_pct=_safe_float(farmer.get("green_coverage_pct")),
        stress_zone_pct=_safe_float(farmer.get("estimated_stress_zone_pct")),
        vigor_score=_safe_float(farmer.get("vegetation_vigor_score")),
        vari=_safe_float(rgb_indices.get("vari")),
        gli=_safe_float(rgb_indices.get("gli")),
        ngrdi=_safe_float(rgb_indices.get("ngrdi")),
        exg=_safe_float(rgb_indices.get("exg")),
        tgi=_safe_float(rgb_indices.get("tgi")),
        mgrvi=_safe_float(rgb_indices.get("mgrvi")),
        lai_proxy=_safe_float(rgb_indices.get("lai_proxy")),
        canopy_cover_pct=_safe_float(veg.get("percent_canopy_cover")),
        biomass_score=_safe_float(veg.get("relative_biomass_score")),
        uniformity_score=_safe_float(veg.get("stand_uniformity_score")),
        yield_potential_pct=_safe_float(veg.get("relative_yield_potential_pct")),
    )


def load_payloads(input_path: Path) -> tuple[list[AnalysisRecord], dict[str, Any] | None, list[dict[str, Any]]]:
    if input_path.is_file():
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        analysis_payload, segments = _extract_analysis_payload(raw)
        if analysis_payload is None:
            raise ValueError(f"Unsupported JSON schema in {input_path}")
        record = _record_from_payload(analysis_payload, input_path.stem)
        return [record] if record else [], analysis_payload, segments

    records: list[AnalysisRecord] = []
    for json_path in sorted(input_path.rglob("*.json")):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidates = raw if isinstance(raw, list) else [raw]
        for item in candidates:
            analysis_payload, _segments = _extract_analysis_payload(item)
            if analysis_payload is None:
                continue
            record = _record_from_payload(analysis_payload, json_path.stem)
            if record is not None:
                records.append(record)
    return records, None, []


def _clean(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def _mean(values: list[float]) -> float:
    clean = _clean(values)
    return float(statistics.fmean(clean)) if clean else float("nan")


def _set_matlabish_style() -> None:
    plt.style.use("classic")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f7",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.autolayout": True,
        }
    )


def plot_single_analysis(record: AnalysisRecord, out_dir: Path) -> Path:
    _set_matlabish_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Single Analysis Overview: {record.image_id}", fontsize=15, fontweight="bold")

    rgb_vals = [record.mean_red, record.mean_green, record.mean_blue]
    axes[0, 0].bar(["Red", "Green", "Blue"], rgb_vals, color=["#d9534f", "#5cb85c", "#428bca"])
    axes[0, 0].set_title("Mean RGB Channels")
    axes[0, 0].set_ylabel("Mean value (0-255)")

    feature_labels = ["Green Cover", "Stress Zone", "Vigor", "Confidence", "Health Score"]
    feature_vals = [
        record.green_coverage_pct,
        record.stress_zone_pct,
        record.vigor_score,
        record.confidence * 100.0,
        record.health_score * 100.0,
    ]
    axes[0, 1].bar(feature_labels, feature_vals, color="#4c72b0")
    axes[0, 1].set_title("Core Farmer Features")
    axes[0, 1].set_ylabel("Percent / score")
    axes[0, 1].tick_params(axis="x", rotation=18)

    index_labels = ["VARI", "GLI", "NGRDI", "ExG", "TGI", "MGRVI", "LAI"]
    index_vals = [record.vari, record.gli, record.ngrdi, record.exg, record.tgi, record.mgrvi, record.lai_proxy]
    axes[0, 2].bar(index_labels, index_vals, color="#55a868")
    axes[0, 2].set_title("RGB Vegetation Indices")
    axes[0, 2].tick_params(axis="x", rotation=18)

    mgmt_labels = ["Canopy", "Biomass", "Uniformity", "Yield Potential"]
    mgmt_vals = [
        record.canopy_cover_pct,
        record.biomass_score,
        record.uniformity_score,
        record.yield_potential_pct,
    ]
    axes[1, 0].bar(mgmt_labels, mgmt_vals, color="#8172b3")
    axes[1, 0].set_title("Derived Agronomic Proxies")
    axes[1, 0].set_ylabel("Percent / relative score")
    axes[1, 0].tick_params(axis="x", rotation=18)

    health_band_score = BAND_TO_SCORE.get(record.health_band, 0.0)
    axes[1, 1].barh(["Band"], [health_band_score], color="#c44e52")
    axes[1, 1].set_xlim(0, 4.2)
    axes[1, 1].set_xticks([0, 1, 2, 3, 4])
    axes[1, 1].set_xticklabels(["N/A", "Critical", "Stressed", "Watch", "Healthy"])
    axes[1, 1].set_title(f"Health Band: {record.health_band.upper()}")

    axes[1, 2].axis("off")
    summary_text = (
        f"Image ID: {record.image_id}\n"
        f"Created: {record.created_at or 'N/A'}\n"
        f"Health Score: {record.health_score:.3f}\n"
        f"Confidence: {record.confidence:.3f}\n"
        f"Green Coverage: {record.green_coverage_pct:.2f}%\n"
        f"Stress Zone: {record.stress_zone_pct:.2f}%\n"
        f"Vigor Score: {record.vigor_score:.2f}\n"
        f"Primary Indexes: VARI={record.vari:.3f}, GLI={record.gli:.3f}, NGRDI={record.ngrdi:.3f}, ExG={record.exg:.3f}"
    )
    axes[1, 2].text(0.0, 0.98, summary_text, va="top", ha="left", family="monospace", fontsize=10)

    out_path = out_dir / f"{record.image_id}_single_analysis.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _segment_grid(segments: list[dict[str, Any]], key: str, fill_value: float = np.nan) -> np.ndarray:
    rows = max(int(seg.get("row", 0)) for seg in segments)
    cols = max(int(seg.get("col", 0)) for seg in segments)
    grid = np.full((rows, cols), fill_value, dtype=float)
    for seg in segments:
        r = int(seg.get("row", 0)) - 1
        c = int(seg.get("col", 0)) - 1
        if r < 0 or c < 0:
            continue
        val = BAND_TO_SCORE.get(str(seg.get("health_band") or "na").lower(), fill_value) if key == "band_score" else _safe_float(seg.get(key), fill_value)
        grid[r, c] = val
    return grid


def plot_segment_heatmaps(segments: list[dict[str, Any]], image_id: str, out_dir: Path) -> Path | None:
    if not segments:
        return None

    _set_matlabish_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Segment Heatmaps: {image_id}", fontsize=15, fontweight="bold")

    heatmap_specs = [
        ("band_score", "Health Band Score", "viridis"),
        ("health_score", "Health Score", "YlGn"),
        ("green_coverage_pct", "Green Coverage %", "Greens"),
        ("estimated_stress_zone_pct", "Stress Zone %", "YlOrRd"),
        ("vegetation_vigor_score", "Vigor Score", "PuBuGn"),
        ("relative_yield_potential_pct", "Yield Potential %", "cividis"),
    ]

    for ax, (key, title, cmap) in zip(axes.flatten(), heatmap_specs):
        grid = _segment_grid(segments, key)
        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.set_xticklabels([str(i + 1) for i in range(grid.shape[1])])
        ax.set_yticklabels([str(i + 1) for i in range(grid.shape[0])])

        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                value = grid[row, col]
                label = "N/A" if math.isnan(value) else f"{value:.1f}"
                ax.text(col, row, label, ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = out_dir / f"{image_id}_segment_heatmaps.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_directory_summary(records: list[AnalysisRecord], out_dir: Path) -> Path:
    _set_matlabish_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Batch Analysis Summary", fontsize=15, fontweight="bold")

    band_counts = [sum(1 for r in records if r.health_band == band) for band in BANDS[:-1]]
    axes[0, 0].bar(BANDS[:-1], band_counts, color="#4c72b0")
    axes[0, 0].set_title("Health Band Distribution")

    health_scores = [r.health_score for r in records]
    confidences = [r.confidence for r in records]
    axes[0, 1].plot(health_scores, marker="o", linewidth=1.5, label="Health Score")
    axes[0, 1].plot(confidences, marker="s", linewidth=1.2, label="Confidence")
    axes[0, 1].set_title("Scores Across Samples")
    axes[0, 1].set_xlabel("Sample index")
    axes[0, 1].legend()

    axes[0, 2].scatter(
        [r.green_coverage_pct for r in records],
        [r.health_score for r in records],
        c=[r.stress_zone_pct for r in records],
        cmap="RdYlGn_r",
        edgecolors="black",
    )
    axes[0, 2].set_title("Health vs Green Coverage")
    axes[0, 2].set_xlabel("Green Coverage %")
    axes[0, 2].set_ylabel("Health Score")

    index_matrix = np.array(
        [
            [_mean([r.vari for r in records]), _mean([r.gli for r in records]), _mean([r.ngrdi for r in records]), _mean([r.exg for r in records]), _mean([r.tgi for r in records]), _mean([r.mgrvi for r in records]), _mean([r.lai_proxy for r in records])]
        ]
    )
    im = axes[1, 0].imshow(index_matrix, cmap="viridis", aspect="auto")
    axes[1, 0].set_title("Mean Vegetation Indices")
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(["VARI", "GLI", "NGRDI", "ExG", "TGI", "MGRVI", "LAI"], rotation=20)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    proxy_names = ["Canopy", "Biomass", "Uniformity", "Yield"]
    proxy_vals = [
        _mean([r.canopy_cover_pct for r in records]),
        _mean([r.biomass_score for r in records]),
        _mean([r.uniformity_score for r in records]),
        _mean([r.yield_potential_pct for r in records]),
    ]
    axes[1, 1].bar(proxy_names, proxy_vals, color="#55a868")
    axes[1, 1].set_title("Mean Agronomic Proxies")

    corr_features = np.array(
        [
            [r.health_score, r.confidence, r.green_coverage_pct, r.stress_zone_pct, r.vigor_score]
            for r in records
        ],
        dtype=float,
    )
    corr_labels = ["Health", "Conf", "Green", "Stress", "Vigor"]
    corr = np.corrcoef(corr_features, rowvar=False)
    corr_im = axes[1, 2].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1, 2].set_title("Feature Correlation Matrix")
    axes[1, 2].set_xticks(range(len(corr_labels)))
    axes[1, 2].set_yticks(range(len(corr_labels)))
    axes[1, 2].set_xticklabels(corr_labels, rotation=25)
    axes[1, 2].set_yticklabels(corr_labels)
    fig.colorbar(corr_im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    out_path = out_dir / "batch_analysis_summary.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RGB vegetation analysis outputs with matplotlib.")
    parser.add_argument("--input", type=Path, required=True, help="Analysis JSON file or directory containing JSON files.")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/plots"), help="Output folder for generated figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, single_payload, segments = load_payloads(args.input)
    if not records:
        raise SystemExit("No supported analysis JSON data found.")

    generated: list[Path] = []
    if single_payload is not None:
        single_record = records[0]
        generated.append(plot_single_analysis(single_record, args.out_dir))
        heatmap_path = plot_segment_heatmaps(segments, single_record.image_id, args.out_dir)
        if heatmap_path is not None:
            generated.append(heatmap_path)
    else:
        generated.append(plot_directory_summary(records, args.out_dir))

    for output in generated:
        print(f"[ok] Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
