#!/usr/bin/env python3
"""
Model evaluation and product-readiness report generator for the RGB vegetation AI model.

Usage:
  python scripts/model_evaluation_report.py \
      --results-root data/results \
      --out-dir analysis/evaluation_report

Optional ground-truth labels CSV (for true classification metrics):
  python scripts/model_evaluation_report.py \
      --results-root data/results \
      --labels-csv data/labels.csv \
      --out-dir analysis/evaluation_report

Expected labels CSV columns:
  image_id,true_band
where true_band is one of: healthy, watch, stressed, critical
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

BANDS = ["healthy", "watch", "stressed", "critical"]
BAND_TO_INDEX = {b: i for i, b in enumerate(BANDS)}


@dataclass
class Record:
    image_id: str
    created_at: str
    health_score: float
    health_band: str
    confidence: float
    green_coverage_pct: float
    stress_zone_pct: float
    vigor_score: float
    vari: float
    gli: float
    ngrdi: float
    exg: float
    tgi: float | None
    mgrvi: float | None
    lai_proxy: float | None


def _safe_float(value: Any, fallback: float = float("nan")) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_records(results_root: Path) -> list[Record]:
    records: list[Record] = []
    for p in results_root.rglob("*.json"):
        if p.name == "results_index.json":
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        report = payload.get("report") or {}
        model_result = report.get("model_result") or {}
        farmer_features = payload.get("farmer_features") or {}
        rgb_indices = payload.get("rgb_indices") or {}
        if not isinstance(model_result, dict):
            continue

        band = str(model_result.get("health_band") or "").strip().lower()
        if band not in BAND_TO_INDEX:
            continue

        record = Record(
            image_id=str(payload.get("image_id") or p.stem),
            created_at=str(payload.get("created_at") or ""),
            health_score=_safe_float(model_result.get("health_score")),
            health_band=band,
            confidence=_safe_float(model_result.get("confidence")),
            green_coverage_pct=_safe_float(farmer_features.get("green_coverage_pct")),
            stress_zone_pct=_safe_float(farmer_features.get("estimated_stress_zone_pct")),
            vigor_score=_safe_float(farmer_features.get("vegetation_vigor_score")),
            vari=_safe_float(rgb_indices.get("vari")),
            gli=_safe_float(rgb_indices.get("gli")),
            ngrdi=_safe_float(rgb_indices.get("ngrdi")),
            exg=_safe_float(rgb_indices.get("exg")),
            tgi=_safe_float(rgb_indices.get("tgi"), fallback=float("nan")),
            mgrvi=_safe_float(rgb_indices.get("mgrvi"), fallback=float("nan")),
            lai_proxy=_safe_float(rgb_indices.get("lai_proxy"), fallback=float("nan")),
        )
        records.append(record)

    records.sort(key=lambda r: r.created_at)
    return records


def load_labels(labels_csv: Path | None) -> dict[str, str]:
    if labels_csv is None or not labels_csv.exists():
        return {}

    labels: dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = str((row.get("image_id") or "")).strip()
            band = str((row.get("true_band") or "")).strip().lower()
            if img and band in BAND_TO_INDEX:
                labels[img] = band
    return labels


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(statistics.fmean(clean)) if clean else float("nan")


def _std(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0


def correlation(x: list[float], y: list[float]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for xv, yv in zip(x, y):
        if math.isnan(xv) or math.isnan(yv):
            continue
        xs.append(xv)
        ys.append(yv)
    if len(xs) < 3:
        return float("nan")
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def build_confusion(records: list[Record], labels: dict[str, str]) -> tuple[np.ndarray, int]:
    m = np.zeros((4, 4), dtype=int)
    n = 0
    for r in records:
        t = labels.get(r.image_id)
        if t is None:
            continue
        m[BAND_TO_INDEX[t], BAND_TO_INDEX[r.health_band]] += 1
        n += 1
    return m, n


def classification_metrics(conf: np.ndarray) -> dict[str, float]:
    total = conf.sum()
    if total == 0:
        return {}

    acc = float(np.trace(conf) / total)

    recalls = []
    precisions = []
    f1s = []
    for i in range(4):
        tp = conf[i, i]
        fn = conf[i, :].sum() - tp
        fp = conf[:, i].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    return {
        "accuracy": acc,
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "macro_precision": float(sum(precisions) / len(precisions)),
        "macro_recall": float(sum(recalls) / len(recalls)),
        "macro_f1": float(sum(f1s) / len(f1s)),
    }


def svg_bar_chart(title: str, labels: list[str], values: list[float], color: str = "#1FA87A") -> str:
    width = 760
    height = 320
    pad = 44
    chart_w = width - 2 * pad
    chart_h = height - 2 * pad

    vmax = max(values) if values else 1.0
    vmax = vmax if vmax > 0 else 1.0
    bar_w = chart_w / max(len(values), 1)

    bars = []
    for i, v in enumerate(values):
        h = (v / vmax) * (chart_h - 20)
        x = pad + i * bar_w + 12
        y = pad + chart_h - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-24:.1f}" height="{h:.1f}" rx="6" fill="{color}" />')
        bars.append(f'<text x="{x + (bar_w-24)/2:.1f}" y="{pad + chart_h + 18:.1f}" text-anchor="middle" fill="#BFCFC3" font-size="12">{labels[i]}</text>')
        bars.append(f'<text x="{x + (bar_w-24)/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" fill="#E6F1EC" font-size="11">{v:.2f}</text>')

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#0B1A14" stroke="#2C3F36"/>
  <text x="20" y="30" fill="#F2FFF9" font-size="18" font-weight="700">{title}</text>
  <line x1="{pad}" y1="{pad + chart_h}" x2="{pad + chart_w}" y2="{pad + chart_h}" stroke="#456157"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + chart_h}" stroke="#456157"/>
  {''.join(bars)}
</svg>
"""


def svg_scatter(title: str, xs: list[float], ys: list[float], xlab: str, ylab: str) -> str:
    width = 760
    height = 320
    pad = 52
    chart_w = width - 2 * pad
    chart_h = height - 2 * pad

    points = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
    if not points:
        return f"<div>No data for {title}</div>"

    xvals = [p[0] for p in points]
    yvals = [p[1] for p in points]
    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)
    if xmax - xmin < 1e-9:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0

    dots = []
    for x, y in points:
        px = pad + ((x - xmin) / (xmax - xmin)) * chart_w
        py = pad + chart_h - ((y - ymin) / (ymax - ymin)) * chart_h
        dots.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#4CD4A5" opacity="0.85"/>')

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#0B1A14" stroke="#2C3F36"/>
  <text x="20" y="30" fill="#F2FFF9" font-size="18" font-weight="700">{title}</text>
  <line x1="{pad}" y1="{pad + chart_h}" x2="{pad + chart_w}" y2="{pad + chart_h}" stroke="#456157"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + chart_h}" stroke="#456157"/>
  {''.join(dots)}
  <text x="{pad + chart_w/2:.1f}" y="{height-10}" text-anchor="middle" fill="#BFCFC3" font-size="12">{xlab}</text>
  <text x="16" y="{pad + chart_h/2:.1f}" text-anchor="middle" fill="#BFCFC3" font-size="12" transform="rotate(-90 16 {pad + chart_h/2:.1f})">{ylab}</text>
</svg>
"""


def svg_confusion(conf: np.ndarray) -> str:
    width = 540
    height = 520
    pad = 84
    cell = 92
    mx = conf.max() if conf.size else 1
    mx = int(mx) if int(mx) > 0 else 1

    cells = []
    for r in range(4):
        for c in range(4):
            v = int(conf[r, c])
            intensity = int(30 + (200 * (v / mx)))
            fill = f"rgb(15,{intensity},110)"
            x = pad + c * cell
            y = pad + r * cell
            cells.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#2C3F36"/>')
            cells.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 5}" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">{v}</text>')

    labels = []
    for i, b in enumerate(BANDS):
        labels.append(f'<text x="{pad + i*cell + cell/2}" y="{pad - 14}" text-anchor="middle" fill="#BFCFC3" font-size="12">{b}</text>')
        labels.append(f'<text x="{pad - 14}" y="{pad + i*cell + cell/2 + 4}" text-anchor="end" fill="#BFCFC3" font-size="12">{b}</text>')

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#0B1A14" stroke="#2C3F36"/>
  <text x="20" y="34" fill="#F2FFF9" font-size="18" font-weight="700">Confusion Matrix (True vs Predicted Band)</text>
  <text x="{pad + 2*cell}" y="{height - 20}" text-anchor="middle" fill="#BFCFC3" font-size="12">Predicted</text>
  <text x="20" y="{pad + 2*cell}" text-anchor="middle" fill="#BFCFC3" font-size="12" transform="rotate(-90 20 {pad + 2*cell})">True</text>
  {''.join(cells)}
  {''.join(labels)}
</svg>
"""


def render_html(
    records: list[Record],
    labels: dict[str, str],
    out_html: Path,
    project_name: str,
) -> None:
    n = len(records)
    if n == 0:
        out_html.write_text("<h1>No analysis records found.</h1>", encoding="utf-8")
        return

    score_vals = [r.health_score for r in records]
    conf_vals = [r.confidence for r in records]
    green_vals = [r.green_coverage_pct for r in records]
    stress_vals = [r.stress_zone_pct for r in records]

    band_counts = [sum(1 for r in records if r.health_band == b) for b in BANDS]

    corr_score_green = correlation(score_vals, green_vals)
    corr_score_stress = correlation(score_vals, stress_vals)

    conf_m, labeled_n = build_confusion(records, labels)
    cls_metrics = classification_metrics(conf_m)

    data_quality_checks = [
        ("Sample count >= 100", n >= 100),
        ("All 4 health bands present", sum(1 for x in band_counts if x > 0) == 4),
        ("Score vs Green coverage positively correlated", (not math.isnan(corr_score_green)) and corr_score_green > 0.15),
        ("Score vs Stress zone negatively correlated", (not math.isnan(corr_score_stress)) and corr_score_stress < -0.15),
    ]
    if labeled_n > 0:
        data_quality_checks.extend(
            [
                ("Labeled accuracy >= 0.70", cls_metrics.get("accuracy", 0.0) >= 0.70),
                ("Macro F1 >= 0.65", cls_metrics.get("macro_f1", 0.0) >= 0.65),
            ]
        )

    readiness_score = round(100.0 * sum(1 for _, ok in data_quality_checks if ok) / len(data_quality_checks), 1)

    bar_svg = svg_bar_chart("Health Band Distribution", BANDS, [float(c) for c in band_counts])
    scatter_green_svg = svg_scatter(
        "Health Score vs Green Coverage",
        score_vals,
        green_vals,
        "Health Score",
        "Green Coverage %",
    )
    scatter_stress_svg = svg_scatter(
        "Health Score vs Stress Zone",
        score_vals,
        stress_vals,
        "Health Score",
        "Stress Zone %",
    )
    conf_svg = svg_confusion(conf_m) if labeled_n > 0 else ""

    checks_html = "".join(
        f"<li><strong>{name}:</strong> {'PASS' if ok else 'FAIL'}</li>" for name, ok in data_quality_checks
    )

    classification_html = ""
    if labeled_n > 0:
        classification_html = f"""
        <h2>Supervised Validation (with Ground Truth)</h2>
        <p>Labeled samples used: <strong>{labeled_n}</strong></p>
        <ul>
          <li>Accuracy: <strong>{cls_metrics.get('accuracy', float('nan')):.3f}</strong></li>
          <li>Balanced Accuracy: <strong>{cls_metrics.get('balanced_accuracy', float('nan')):.3f}</strong></li>
          <li>Macro Precision: <strong>{cls_metrics.get('macro_precision', float('nan')):.3f}</strong></li>
          <li>Macro Recall: <strong>{cls_metrics.get('macro_recall', float('nan')):.3f}</strong></li>
          <li>Macro F1: <strong>{cls_metrics.get('macro_f1', float('nan')):.3f}</strong></li>
        </ul>
        {conf_svg}
        """
    else:
        classification_html = """
        <h2>Supervised Validation (Ground Truth Missing)</h2>
        <p>No labels CSV provided. This report currently shows unsupervised/proxy evaluation only.
        For product defense, add independent field labels and rerun with <code>--labels-csv</code>.</p>
        """

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AI Model Evaluation Report</title>
  <style>
    body {{ background:#07120F; color:#E6F1EC; font-family:Inter,Segoe UI,Arial,sans-serif; margin:0; padding:24px; }}
    .wrap {{ max-width:1200px; margin:0 auto; }}
    .card {{ background:#0B1A14; border:1px solid #2C3F36; border-radius:14px; padding:16px; margin:0 0 14px 0; }}
    h1,h2,h3 {{ margin:0 0 10px 0; }}
    p,li {{ color:#CFE0D8; line-height:1.45; }}
    code {{ background:#10261E; padding:2px 6px; border-radius:6px; color:#8CFBD8; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .kpi {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:10px; }}
    .kpi .box {{ background:#10261E; border:1px solid #2C3F36; border-radius:10px; padding:10px; }}
    .big {{ font-size:22px; font-weight:800; color:#8CFBD8; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{project_name} - RGB AI Model Evaluation & Product Readiness</h1>
      <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      <div class="kpi">
        <div class="box"><div>Samples</div><div class="big">{n}</div></div>
        <div class="box"><div>Readiness Score</div><div class="big">{readiness_score}%</div></div>
        <div class="box"><div>Mean Health Score</div><div class="big">{_mean(score_vals):.3f}</div></div>
        <div class="box"><div>Mean Confidence</div><div class="big">{_mean(conf_vals):.3f}</div></div>
      </div>
    </div>

    <div class="card">
      <h2>How The Model Is Computed (Transparent Processing)</h2>
      <ol>
        <li>Image RGB channels are summarized into <code>mean_red</code>, <code>mean_green</code>, <code>mean_blue</code>, and green coverage.</li>
        <li>Vegetation proxies are computed: <code>VARI</code>, <code>GLI</code>, <code>NGRDI</code>, <code>ExG</code>.</li>
        <li>Weighted score model combines normalized proxies: VARI(0.30), GLI(0.25), NGRDI(0.20), ExG(0.25).</li>
        <li>Context penalties adjust score (low rainfall / extreme temperature).</li>
        <li>Score is mapped to health bands: healthy/watch/stressed/critical.</li>
        <li>Confidence is derived from proxy spread and bounded by model limits.</li>
      </ol>
      <p>Extended indices used for operations planning in this project: <code>TGI</code>, <code>MGRVI</code>, and <code>LAI proxy</code> plus canopy/stress/biomass/yield proxies.</p>
    </div>

    <div class="card">
      <h2>Core Distribution & Relationship Metrics</h2>
      <ul>
        <li>Health score std-dev: <strong>{_std(score_vals):.3f}</strong></li>
        <li>Confidence std-dev: <strong>{_std(conf_vals):.3f}</strong></li>
        <li>Correlation(score, green coverage): <strong>{corr_score_green:.3f}</strong></li>
        <li>Correlation(score, stress zone): <strong>{corr_score_stress:.3f}</strong></li>
      </ul>
      {bar_svg}
    </div>

    <div class="grid">
      <div class="card">{scatter_green_svg}</div>
      <div class="card">{scatter_stress_svg}</div>
    </div>

    <div class="card">
      {classification_html}
    </div>

    <div class="card">
      <h2>Product Defense Checklist (Go/No-Go Gates)</h2>
      <ul>{checks_html}</ul>
      <p><strong>Interpretation:</strong> This model is operationally useful when it shows stable correlations, diverse band coverage,
      and (when labels exist) acceptable supervised accuracy/F1. For production claims, keep a labeled validation set,
      monitor drift monthly, and audit recommendations against field outcomes.</p>
    </div>

    <div class="card">
      <h2>Risks, Limits, and Mitigations</h2>
      <ul>
        <li><strong>RGB-only limitation:</strong> This is a proxy to multispectral indices; mitigate by field scouting in critical zones.</li>
        <li><strong>Lighting sensitivity:</strong> Enforce consistent capture time and weather windows.</li>
        <li><strong>Domain shift:</strong> Different crop stages and camera settings can move score distributions.</li>
        <li><strong>Recommendation safety:</strong> Treat recommendations as decision support, not autonomous farm control.</li>
      </ul>
    </div>
  </div>
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation and product-readiness report for RGB AI model.")
    parser.add_argument("--results-root", type=Path, default=Path("data/results"), help="Root folder of analysis JSON files.")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional CSV with image_id,true_band.")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/evaluation_report"), help="Output directory.")
    parser.add_argument("--project-name", type=str, default="Farm RGB Model", help="Display name in report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.results_root)
    labels = load_labels(args.labels_csv)

    out_html = args.out_dir / "model_evaluation_report.html"
    render_html(records, labels, out_html, args.project_name)

    print(f"[ok] Report generated: {out_html}")
    print(f"[ok] Records used: {len(records)}")
    print(f"[ok] Labels used: {len(labels)}")
    if len(labels) == 0:
        print("[note] No labels CSV provided; supervised metrics section is limited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
