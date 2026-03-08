import os
import json
import uuid
import mimetypes
import io
import base64
import textwrap
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask import Response
from dotenv import load_dotenv
from supabase import Client, create_client
import numpy as np
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from analysis import FieldContext, RGBSnapshot, interpret_field_from_rgb, rgb_to_vegetation_proxies

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "replace-this-in-production")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit for stability on small instances.

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://sqjowujqqtljmwuizgqv.supabase.co")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
)
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "user_info")
SUPABASE_IMAGES_TABLE = os.getenv("SUPABASE_IMAGES_TABLE", "uploaded_images")
SUPABASE_RESULTS_TABLE = os.getenv("SUPABASE_RESULTS_TABLE", "analysis_results")
SUPABASE_BUCKET_ORIGINAL = os.getenv("SUPABASE_BUCKET_ORIGINAL", "original-images")
SUPABASE_BUCKET_HEATZONE = os.getenv("SUPABASE_BUCKET_HEATZONE", "ai-images")
SUPABASE_BUCKET_REPORTS = os.getenv("SUPABASE_BUCKET_REPORTS", "reports")

supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_KEY else None
TEMP_ROOT = BASE_DIR / "tmp"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
ANALYSIS_MAX_DIMENSION = 1600


def current_user():
    return session.get("user")


def require_auth():
    if not current_user():
        return False
    return True


def to_upper_text(value: str, fallback: str) -> str:
    text = (value or fallback).strip()
    return text.upper() if text else fallback


def verify_password(stored_password: str, provided_password: str) -> bool:
    # Supports both werkzeug hashes and legacy plain-text rows.
    if not stored_password:
        return False
    if stored_password == provided_password:
        return True
    return check_password_hash(stored_password, provided_password)


def is_allowed_image(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def add_upload_log(user_id: str, message: str, level: str) -> None:
    logs = session.get("upload_logs", [])
    logs.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "message": message,
            "level": level,
        },
    )
    session["upload_logs"] = logs[:100]


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def upload_file_to_bucket(local_path: Path, bucket: str, object_path: str) -> tuple[bool, str]:
    if supabase is None:
        return False, "Supabase client not initialized."
    try:
        content_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        payload = local_path.read_bytes()
        supabase.storage.from_(bucket).upload(
            object_path,
            payload,
            {"content-type": content_type, "upsert": "true"},
        )
        return True, ""
    except Exception as exc:
        print(f"[storage] upload failed bucket={bucket} path={object_path} error={exc}")
        return False, str(exc)


def upload_json_to_bucket(bucket: str, object_path: str, payload: dict[str, Any]) -> tuple[bool, str]:
    if supabase is None:
        return False, "Supabase client not initialized."
    try:
        supabase.storage.from_(bucket).upload(
            object_path,
            json.dumps(payload, indent=2).encode("utf-8"),
            {"content-type": "application/json", "upsert": "true"},
        )
        return True, ""
    except Exception as exc:
        print(f"[storage] json upload failed bucket={bucket} path={object_path} error={exc}")
        return False, str(exc)


def signed_or_local_url(
    bucket: str,
    object_path: str | None,
    fallback_url: str,
    expires_in: int = 3600,
) -> str:
    if supabase is None or not object_path:
        return fallback_url
    try:
        signed = supabase.storage.from_(bucket).create_signed_url(object_path, expires_in)
        if isinstance(signed, dict):
            for k in ("signedURL", "signedUrl", "signed_url"):
                if signed.get(k):
                    return signed[k]
    except Exception:
        pass
    return fallback_url


def download_file_from_bucket(bucket: str, object_path: str, destination: Path) -> bool:
    if supabase is None:
        return False
    try:
        payload = supabase.storage.from_(bucket).download(object_path)
        if payload is None:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return True
    except Exception:
        return False


def delete_file_from_bucket(bucket: str, object_path: str) -> bool:
    if supabase is None or not object_path:
        return False
    try:
        supabase.storage.from_(bucket).remove([object_path])
        return True
    except Exception:
        return False


def default_result_personalization() -> dict[str, Any]:
    return {
        "title": "",
        "field_name": "",
        "crop_type": "",
        "farmer_notes": "",
        "recommendation_checks": [],
        "flags": [],
    }


def normalize_result_personalization(data: dict[str, Any] | None) -> dict[str, Any]:
    base = default_result_personalization()
    if not isinstance(data, dict):
        return base

    base["title"] = str(data.get("title", "") or "")[:120]
    base["field_name"] = str(data.get("field_name", "") or "")[:120]
    base["crop_type"] = str(data.get("crop_type", "") or "")[:120]
    base["farmer_notes"] = str(data.get("farmer_notes", "") or "")[:3000]
    base["recommendation_checks"] = [bool(v) for v in (data.get("recommendation_checks") or [])[:64]]

    flags: list[dict[str, Any]] = []
    for raw in data.get("flags") or []:
        if not isinstance(raw, dict):
            continue
        try:
            fx = float(raw.get("x", 0))
            fy = float(raw.get("y", 0))
        except (TypeError, ValueError):
            continue
        if fx < 0 or fx > 1 or fy < 0 or fy > 1:
            continue
        flags.append(
            {
                "id": str(raw.get("id") or str(uuid.uuid4())),
                "x": round(fx, 5),
                "y": round(fy, 5),
                "label": str(raw.get("label", "") or "Flag")[:120],
            }
        )
    base["flags"] = flags[:150]
    return base


def read_result_personalization(analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(analysis, dict):
        return default_result_personalization()
    return normalize_result_personalization((analysis.get("personalization") or {}))


def save_result_personalization(user_id: str, image_id: str, personalization: dict[str, Any]) -> tuple[bool, str]:
    if supabase is None:
        return False, "Supabase client not initialized."

    try:
        result_resp = (
            supabase.table(SUPABASE_RESULTS_TABLE)
            .select("image_id, analysis_json")
            .eq("image_id", image_id)
            .limit(1)
            .execute()
        )
        rows = result_resp.data or []
        if not rows:
            return False, "Result not found."

        current_analysis = rows[0].get("analysis_json") or {}
        if not isinstance(current_analysis, dict):
            current_analysis = {}
        current_analysis["personalization"] = normalize_result_personalization(personalization)
        current_analysis["updated_at"] = now_utc_iso()

        supabase.table(SUPABASE_RESULTS_TABLE).update(
            {
                "analysis_json": current_analysis,
                "updated_at": now_utc_iso(),
            }
        ).eq("image_id", image_id).execute()

        report_object_path = f"{user_id}/{image_id}.json"
        ok_json, err_json = upload_json_to_bucket(SUPABASE_BUCKET_REPORTS, report_object_path, current_analysis)
        require_bucket_upload(ok_json, "report json", err_json)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def upsert_result_in_supabase(user_id: str, entry: dict[str, Any], analysis_payload: dict[str, Any]) -> bool:
    if supabase is None:
        return False
    try:
        supabase.table(SUPABASE_IMAGES_TABLE).upsert(
            {
                "image_id": entry["image_id"],
                "user_id": user_id,
                "filename": entry["filename"],
                "original_path": entry["original_storage_path"],
                "heatzone_path": entry["heatzone_storage_path"],
                "status": "processed",
                "uploaded_at": entry["created_at"],
                "updated_at": entry["updated_at"],
            },
            on_conflict="image_id",
        ).execute()

        supabase.table(SUPABASE_RESULTS_TABLE).upsert(
            {
                "result_id": entry["image_id"],
                "image_id": entry["image_id"],
                "analysis_json": analysis_payload,
                "summary": entry["summary"],
                "health_band": entry["health_band"],
                "health_score": entry["health_score"],
                "confidence": entry["confidence"],
                "created_at": entry["created_at"],
                "updated_at": entry["updated_at"],
            },
            on_conflict="image_id",
        ).execute()
        return True
    except Exception:
        return False


def require_bucket_upload(ok: bool, label: str, detail: str = "") -> None:
    if not ok:
        extra = f" ({detail})" if detail else ""
        raise RuntimeError(f"Supabase bucket upload failed for {label}{extra}.")


def load_user_results_from_supabase(user_id: str) -> list[dict[str, Any]] | None:
    if supabase is None:
        return None
    try:
        images_resp = (
            supabase.table(SUPABASE_IMAGES_TABLE)
            .select("image_id, filename, original_path, heatzone_path, uploaded_at, updated_at, status")
            .eq("user_id", user_id)
            .order("uploaded_at", desc=True)
            .execute()
        )
        image_rows = images_resp.data or []
        if not image_rows:
            return []

        image_ids = [row["image_id"] for row in image_rows]
        results_resp = (
            supabase.table(SUPABASE_RESULTS_TABLE)
            .select("image_id, analysis_json, summary, health_band, health_score, confidence, created_at, updated_at")
            .in_("image_id", image_ids)
            .execute()
        )
        result_rows = {row["image_id"]: row for row in (results_resp.data or [])}

        merged: list[dict[str, Any]] = []
        for row in image_rows:
            res = result_rows.get(row["image_id"], {})
            original_ref = row.get("original_path", "")
            heatzone_ref = row.get("heatzone_path", "")
            if not original_ref or not heatzone_ref:
                continue
            personalization = read_result_personalization(res.get("analysis_json"))
            display_title = personalization.get("title") or f"Result {row['image_id']}"
            entry = {
                "image_id": row["image_id"],
                "filename": row.get("filename", ""),
                "created_at": row.get("uploaded_at") or res.get("created_at") or "",
                "updated_at": row.get("updated_at") or res.get("updated_at") or "",
                "original_url": signed_or_local_url(SUPABASE_BUCKET_ORIGINAL, original_ref, ""),
                "heatzone_url": signed_or_local_url(SUPABASE_BUCKET_HEATZONE, heatzone_ref, ""),
                "original_storage_path": original_ref,
                "heatzone_storage_path": heatzone_ref,
                "analysis_json": res.get("analysis_json"),
                "summary": res.get("summary") or "Analysis pending",
                "health_band": res.get("health_band") or "watch",
                "health_score": res.get("health_score") or 0.0,
                "confidence": res.get("confidence") or 0.0,
                "analysis_json_path": "",
                "display_title": display_title,
            }
            if not entry["original_url"] or not entry["heatzone_url"]:
                continue
            merged.append(entry)
        return merged
    except Exception:
        return None


def ensure_temp_dir(user_id: str) -> Path:
    user_temp_dir = TEMP_ROOT / user_id
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    return user_temp_dir


def load_user_results_index(user_id: str) -> list[dict[str, Any]]:
    remote = load_user_results_from_supabase(user_id)
    if remote is not None:
        return remote
    return []


def save_user_results_index(user_id: str, items: list[dict[str, Any]]) -> None:
    # Bucket-only mode: no local persistence.
    return


def create_heatzone_image(original_path: Path, output_path: Path) -> tuple[float, float]:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("Image processing dependency is missing. Install Pillow or OpenCV to enable analysis.") from exc

        raw = np.fromfile(str(original_path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Unable to decode image file.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        score = (g - r) / (r + g + b + 1e-6)

        overlay = np.zeros_like(rgb, dtype=np.uint8)
        mask_critical = score < 0.02
        mask_low = (score >= 0.02) & (score < 0.08)
        mask_watch = (score >= 0.08) & (score < 0.15)
        mask_good = score >= 0.15

        overlay[mask_critical] = (255, 64, 64)
        overlay[mask_low] = (255, 171, 64)
        overlay[mask_watch] = (255, 227, 84)
        overlay[mask_good] = (80, 214, 114)

        blended = (0.55 * rgb + 0.45 * overlay).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        ext = output_path.suffix or ".png"
        ok, encoded = cv2.imencode(ext, out_bgr)
        if not ok:
            raise RuntimeError("Unable to encode heatzone image.")
        encoded.tofile(str(output_path))

        return float(mask_critical.mean()), float(score.mean())

    with Image.open(original_path) as im:
        rgb = im.convert("RGB")
        if max(rgb.size) > ANALYSIS_MAX_DIMENSION:
            rgb.thumbnail((ANALYSIS_MAX_DIMENSION, ANALYSIS_MAX_DIMENSION), Image.Resampling.LANCZOS)

        rgb_np = np.asarray(rgb, dtype=np.float32)
        r = rgb_np[:, :, 0]
        g = rgb_np[:, :, 1]
        b = rgb_np[:, :, 2]
        score = (g - r) / (r + g + b + 1e-6)

        overlay = np.zeros_like(rgb_np, dtype=np.uint8)
        mask_critical = score < 0.02
        mask_low = (score >= 0.02) & (score < 0.08)
        mask_watch = (score >= 0.08) & (score < 0.15)
        mask_good = score >= 0.15

        overlay[mask_critical] = (255, 64, 64)
        overlay[mask_low] = (255, 171, 64)
        overlay[mask_watch] = (255, 227, 84)
        overlay[mask_good] = (80, 214, 114)

        blended_np = (0.55 * rgb_np + 0.45 * overlay).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended_np, mode="RGB").save(output_path, format="PNG")

        return float(mask_critical.mean()), float(score.mean())


def run_rgb_analysis(original_path: Path, user_id: str, image_id: str) -> dict[str, Any]:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("Image processing dependency is missing. Install Pillow or OpenCV.") from exc

        raw = np.fromfile(str(original_path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Unable to decode uploaded image.")
        rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        r = rgb_np[:, :, 0]
        g = rgb_np[:, :, 1]
        b = rgb_np[:, :, 2]
        mean_r = float(r.mean())
        mean_g = float(g.mean())
        mean_b = float(b.mean())
        green_coverage = float(((g > r * 1.05) & (g > b * 1.05)).mean())
    else:
        with Image.open(original_path) as im:
            rgb = im.convert("RGB")
            if max(rgb.size) > ANALYSIS_MAX_DIMENSION:
                rgb.thumbnail((ANALYSIS_MAX_DIMENSION, ANALYSIS_MAX_DIMENSION), Image.Resampling.LANCZOS)

            rgb_np = np.asarray(rgb, dtype=np.float32)
            r = rgb_np[:, :, 0]
            g = rgb_np[:, :, 1]
            b = rgb_np[:, :, 2]
            mean_r = float(r.mean())
            mean_g = float(g.mean())
            mean_b = float(b.mean())
            green_coverage = float(((g > r * 1.05) & (g > b * 1.05)).mean())

    snapshot = RGBSnapshot(
        mean_red=mean_r,
        mean_green=mean_g,
        mean_blue=mean_b,
        green_coverage=green_coverage,
    )
    context = FieldContext(crop_name="Farm Crop", growth_stage="unknown", rainfall_last_7d_mm=0.0, avg_temp_c=0.0)
    report = interpret_field_from_rgb(snapshot, context)
    extended_indices = _compute_extended_rgb_indices(mean_r, mean_g, mean_b, green_coverage)

    temp_dir = ensure_temp_dir(user_id)
    heatzone_filename = f"{image_id}_heatzone.png"
    heatzone_path = temp_dir / heatzone_filename
    stress_ratio, avg_vigor = create_heatzone_image(original_path, heatzone_path)
    zone = _management_zone_recommendation(extended_indices, stress_ratio=stress_ratio)

    analysis_payload = {
        "analysis_version": "rgb-proxy-v1",
        "image_id": image_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "mean_red": round(mean_r, 3),
            "mean_green": round(mean_g, 3),
            "mean_blue": round(mean_b, 3),
            "green_coverage": round(green_coverage, 4),
        },
        "rgb_indices": {
            "vari": round(extended_indices["vari"], 4),
            "gli": round(extended_indices["gli"], 4),
            "ngrdi": round(extended_indices["ngrdi"], 4),
            "exg": round(extended_indices["exg"], 4),
            "tgi": round(extended_indices["tgi"], 4),
            "mgrvi": round(extended_indices["mgrvi"], 4),
            "lai_proxy": round(extended_indices["lai_proxy"], 3),
        },
        "farmer_features": {
            "green_coverage_pct": round(green_coverage * 100, 2),
            "estimated_stress_zone_pct": round(stress_ratio * 100, 2),
            "vegetation_vigor_score": round((avg_vigor + 1) * 50, 2),
            "recommended_scan_schedule": "Rescan in 5-7 days under similar daylight.",
        },
        "vegetation_indices_analysis": {
            "management_zone": zone["zone"],
            "management_action": zone["action"],
            "percent_canopy_cover": round(extended_indices["canopy_cover_pct"], 2),
            "relative_biomass_score": round(extended_indices["relative_biomass_score"], 2),
            "stand_uniformity_score": round(extended_indices["stand_uniformity_score"], 2),
            "relative_yield_potential_pct": round(extended_indices["relative_yield_potential_pct"], 2),
        },
        "report": asdict(report),
    }

    return {
        "analysis": analysis_payload,
        "heatzone_path": str(heatzone_path),
        "heatzone_filename": heatzone_filename,
    }


def _masked_rgb_metrics(rgb_image: Any, mask_image: Any | None = None) -> dict[str, float]:
    pixels = list(rgb_image.getdata())
    if mask_image is not None:
        mask_pixels = list(mask_image.convert("L").getdata())
    else:
        mask_pixels = [255] * len(pixels)

    valid_count = 0
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    green_pixels = 0
    stress_pixels = 0
    vigor_sum = 0.0

    for (r, g, b), m in zip(pixels, mask_pixels):
        if m <= 0:
            continue
        valid_count += 1
        sum_r += r
        sum_g += g
        sum_b += b
        if g > r * 1.05 and g > b * 1.05:
            green_pixels += 1
        score = (g - r) / (r + g + b + 1e-6)
        vigor_sum += score
        if score < 0.02:
            stress_pixels += 1

    if valid_count <= 0:
        raise RuntimeError("Crop area is empty. Draw a larger crop region.")

    mean_r = sum_r / valid_count
    mean_g = sum_g / valid_count
    mean_b = sum_b / valid_count
    green_coverage = green_pixels / valid_count
    stress_ratio = stress_pixels / valid_count
    avg_vigor = vigor_sum / valid_count

    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "green_coverage": green_coverage,
        "stress_ratio": stress_ratio,
        "avg_vigor": avg_vigor,
        "valid_count": float(valid_count),
    }


def _compute_extended_rgb_indices(mean_r: float, mean_g: float, mean_b: float, green_coverage: float) -> dict[str, float]:
    eps = 1e-8
    r = max(mean_r, 0.0) / 255.0
    g = max(mean_g, 0.0) / 255.0
    b = max(mean_b, 0.0) / 255.0

    vari = (g - r) / (g + r - b + eps)
    gli = (2 * g - r - b) / (2 * g + r + b + eps)
    ngrdi = (g - r) / (g + r + eps)
    exg = 2 * g - r - b
    tgi = -0.5 * ((670 - 480) * (r - g) - (670 - 550) * (r - b))
    mgrvi = ((g * g) - (r * r)) / ((g * g) + (r * r) + eps)

    canopy_cover_pct = max(0.0, min(100.0, (0.6 * green_coverage + 0.4 * max(gli, 0.0)) * 100.0))
    lai_proxy = max(0.0, min(6.0, 0.25 + (green_coverage * 4.5) + (max(exg, 0.0) * 1.8)))
    relative_biomass_score = max(0.0, min(100.0, (0.55 * max(vari, 0.0) + 0.45 * (lai_proxy / 6.0)) * 100.0))
    stand_uniformity_score = max(0.0, min(100.0, canopy_cover_pct * 0.85 + max(gli, 0.0) * 15.0))
    relative_yield_potential_pct = max(0.0, min(100.0, (mgrvi + 1.0) * 50.0))

    return {
        "vari": vari,
        "gli": gli,
        "ngrdi": ngrdi,
        "exg": exg,
        "tgi": tgi,
        "mgrvi": mgrvi,
        "lai_proxy": lai_proxy,
        "canopy_cover_pct": canopy_cover_pct,
        "relative_biomass_score": relative_biomass_score,
        "stand_uniformity_score": stand_uniformity_score,
        "relative_yield_potential_pct": relative_yield_potential_pct,
    }


def _management_zone_recommendation(indices: dict[str, float], stress_ratio: float) -> dict[str, str]:
    tgi = float(indices.get("tgi", 0.0))
    vari = float(indices.get("vari", 0.0))
    canopy = float(indices.get("canopy_cover_pct", 0.0))
    yield_potential = float(indices.get("relative_yield_potential_pct", 0.0))

    if tgi < 0.0 or vari < 0.05 or stress_ratio > 0.35:
        zone = "Zone A - Priority Nitrogen & Water"
        action = "Investigate irrigation distribution and apply targeted nitrogen in weak blocks."
    elif canopy < 45.0 or yield_potential < 45.0:
        zone = "Zone B - Monitor and Stabilize"
        action = "Check stand gaps and moisture consistency; monitor growth over the next 3-5 days."
    else:
        zone = "Zone C - Maintain Current Program"
        action = "Current vigor is stable. Continue routine irrigation and nutrition monitoring."

    return {"zone": zone, "action": action}


def _segment_recommendation(segment: dict[str, Any]) -> str:
    if segment.get("empty"):
        return "No valid crop pixels detected in this segment."

    band = str(segment.get("health_band", "watch")).lower()
    canopy = float(segment.get("canopy_cover_pct", 0.0))
    tgi = float(segment.get("tgi", 0.0))
    stress = float(segment.get("estimated_stress_zone_pct", 0.0))

    if band in {"critical", "stressed"} or stress >= 35:
        return "High-risk segment: prioritize field visit, nitrogen correction, and irrigation check."
    if canopy < 45 or tgi < 0:
        return "Low canopy/chlorophyll signal: monitor for thin stands and apply corrective feeding as needed."
    return "Segment is stable. Maintain current management and continue periodic scans."


def _segment_possible_issue(segment: dict[str, Any]) -> str:
    if segment.get("empty"):
        return "No reliable crop signal in this cell (possibly non-crop area or outside selected region)."

    band = str(segment.get("health_band", "watch")).lower()
    stress = float(segment.get("estimated_stress_zone_pct", 0.0))
    canopy = float(segment.get("canopy_cover_pct", 0.0))
    tgi = float(segment.get("tgi", 0.0))

    if band in {"critical", "stressed"} or stress >= 35:
        return "Possible water stress or uneven irrigation, with potential nutrient deficiency in this section."
    if canopy < 45:
        return "Possible sparse stand or early growth delay compared to surrounding segments."
    if tgi < 0:
        return "Possible chlorophyll/nitrogen limitation indicated by weak greenness response."
    return "No major issue detected; segment appears relatively stable."


def _build_masked_heatzone_image(rgb_image: Any, mask_image: Any | None = None) -> Any:
    from PIL import Image

    rgb = rgb_image.convert("RGB")
    pixels = list(rgb.getdata())
    mask_pixels = list(mask_image.convert("L").getdata()) if mask_image is not None else [255] * len(pixels)
    out = Image.new("RGBA", rgb.size)

    rendered: list[tuple[int, int, int, int]] = []
    for (r, g, b), m in zip(pixels, mask_pixels):
        if m <= 0:
            rendered.append((0, 0, 0, 0))
            continue

        score = (g - r) / (r + g + b + 1e-6)
        if score < 0.02:
            overlay = (255, 64, 64)
        elif score < 0.08:
            overlay = (255, 171, 64)
        elif score < 0.15:
            overlay = (255, 227, 84)
        else:
            overlay = (80, 214, 114)

        blended = (
            int(0.55 * r + 0.45 * overlay[0]),
            int(0.55 * g + 0.45 * overlay[1]),
            int(0.55 * b + 0.45 * overlay[2]),
            255,
        )
        rendered.append(blended)

    out.putdata(rendered)
    return out


def _pil_image_to_data_url(image_obj: Any, fmt: str = "PNG") -> str:
    buff = io.BytesIO()
    image_obj.save(buff, format=fmt)
    encoded = base64.b64encode(buff.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _normalize_polygon_points(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    normalized: list[tuple[float, float]] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        try:
            x = float(p.get("x"))
            y = float(p.get("y"))
        except (TypeError, ValueError):
            continue
        if x < 0 or x > 1 or y < 0 or y > 1:
            continue
        normalized.append((round(x, 6), round(y, 6)))
    return normalized


def _resolve_original_result_image(user_id: str, image_id: str, entry: dict[str, Any]) -> tuple[Path | None, bool]:
    original_url = entry.get("original_url", "")
    original_path: Path | None = None
    if isinstance(original_url, str) and original_url.startswith("/") and len(original_url.strip()) > 1:
        original_path = BASE_DIR / original_url.lstrip("/")
    created_temp = False

    if original_path is not None and original_path.exists() and original_path.is_file():
        return original_path, created_temp

    storage_path = entry.get("original_storage_path")
    if not storage_path:
        return None, created_temp

    user_result_img_dir = ensure_temp_dir(str(user_id))
    fallback_name = entry.get("filename") or f"{image_id}.jpg"
    original_path = user_result_img_dir / f"source_{fallback_name}"
    if not download_file_from_bucket(SUPABASE_BUCKET_ORIGINAL, storage_path, original_path):
        return None, created_temp
    created_temp = True
    return original_path, created_temp


def analyze_freeform_cropped_segments(
    original_path: Path,
    image_id: str,
    points: list[tuple[float, float]],
    grid_rows: int,
    grid_cols: int,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    with Image.open(original_path) as source:
        rgb = source.convert("RGB")

    width, height = rgb.size
    px_points: list[tuple[int, int]] = []
    for x_norm, y_norm in points:
        px = min(max(int(round(x_norm * (width - 1))), 0), width - 1)
        py = min(max(int(round(y_norm * (height - 1))), 0), height - 1)
        px_points.append((px, py))

    if len(px_points) < 3:
        raise RuntimeError("Need at least 3 crop points to run advanced analysis.")

    polygon_mask = Image.new("L", rgb.size, 0)
    draw = ImageDraw.Draw(polygon_mask)
    draw.polygon(px_points, fill=255)
    bbox = polygon_mask.getbbox()
    if not bbox:
        raise RuntimeError("Invalid crop selection. Please draw a larger area.")

    crop_rgb = rgb.crop(bbox)
    crop_mask = polygon_mask.crop(bbox)

    masked_original = Image.new("RGBA", crop_rgb.size, (0, 0, 0, 0))
    masked_original.paste(crop_rgb.convert("RGBA"), mask=crop_mask)

    metrics = _masked_rgb_metrics(crop_rgb, crop_mask)
    snapshot = RGBSnapshot(
        mean_red=metrics["mean_r"],
        mean_green=metrics["mean_g"],
        mean_blue=metrics["mean_b"],
        green_coverage=metrics["green_coverage"],
    )
    context = FieldContext(crop_name="Farm Crop", growth_stage="unknown", rainfall_last_7d_mm=0.0, avg_temp_c=0.0)
    report = interpret_field_from_rgb(snapshot, context)
    extended_indices = _compute_extended_rgb_indices(
        metrics["mean_r"], metrics["mean_g"], metrics["mean_b"], metrics["green_coverage"]
    )
    zone = _management_zone_recommendation(extended_indices, stress_ratio=metrics["stress_ratio"])

    heatzone = _build_masked_heatzone_image(crop_rgb, crop_mask)

    crop_width, crop_height = crop_rgb.size
    cell_w = max(crop_width // grid_cols, 1)
    cell_h = max(crop_height // grid_rows, 1)
    segments: list[dict[str, Any]] = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            x0 = col * cell_w
            y0 = row * cell_h
            x1 = crop_width if col == grid_cols - 1 else (col + 1) * cell_w
            y1 = crop_height if row == grid_rows - 1 else (row + 1) * cell_h
            if x1 <= x0 or y1 <= y0:
                continue

            region_rgb = crop_rgb.crop((x0, y0, x1, y1))
            region_mask = crop_mask.crop((x0, y0, x1, y1))
            try:
                region_metrics = _masked_rgb_metrics(region_rgb, region_mask)
                region_snapshot = RGBSnapshot(
                    mean_red=region_metrics["mean_r"],
                    mean_green=region_metrics["mean_g"],
                    mean_blue=region_metrics["mean_b"],
                    green_coverage=region_metrics["green_coverage"],
                )
                region_report = interpret_field_from_rgb(region_snapshot, context)
                region_indices = _compute_extended_rgb_indices(
                    region_metrics["mean_r"],
                    region_metrics["mean_g"],
                    region_metrics["mean_b"],
                    region_metrics["green_coverage"],
                )
                region_zone = _management_zone_recommendation(region_indices, stress_ratio=region_metrics["stress_ratio"])
                health = asdict(region_report)["model_result"]
                segment_payload = {
                    "segment_id": f"r{row + 1}c{col + 1}",
                    "row": row + 1,
                    "col": col + 1,
                    "health_band": health.get("health_band"),
                    "health_score": round(float(health.get("health_score", 0.0)), 2),
                    "confidence": round(float(health.get("confidence", 0.0)), 3),
                    "green_coverage_pct": round(region_metrics["green_coverage"] * 100, 2),
                    "estimated_stress_zone_pct": round(region_metrics["stress_ratio"] * 100, 2),
                    "vegetation_vigor_score": round((region_metrics["avg_vigor"] + 1) * 50, 2),
                    "canopy_cover_pct": round(region_indices["canopy_cover_pct"], 2),
                    "relative_biomass_score": round(region_indices["relative_biomass_score"], 2),
                    "stand_uniformity_score": round(region_indices["stand_uniformity_score"], 2),
                    "relative_yield_potential_pct": round(region_indices["relative_yield_potential_pct"], 2),
                    "tgi": round(region_indices["tgi"], 4),
                    "vari": round(region_indices["vari"], 4),
                    "gli": round(region_indices["gli"], 4),
                    "ngrdi": round(region_indices["ngrdi"], 4),
                    "exg": round(region_indices["exg"], 4),
                    "mgrvi": round(region_indices["mgrvi"], 4),
                    "lai_proxy": round(region_indices["lai_proxy"], 3),
                    "management_zone": region_zone["zone"],
                    "management_action": region_zone["action"],
                }
                segment_payload["recommendation"] = _segment_recommendation(segment_payload)
                segment_payload["possible_issue"] = _segment_possible_issue(segment_payload)
                segments.append(segment_payload)
            except RuntimeError:
                segments.append(
                    {
                        "segment_id": f"r{row + 1}c{col + 1}",
                        "row": row + 1,
                        "col": col + 1,
                        "health_band": "na",
                        "health_score": 0.0,
                        "confidence": 0.0,
                        "green_coverage_pct": 0.0,
                        "estimated_stress_zone_pct": 0.0,
                        "vegetation_vigor_score": 0.0,
                        "canopy_cover_pct": 0.0,
                        "relative_biomass_score": 0.0,
                        "stand_uniformity_score": 0.0,
                        "relative_yield_potential_pct": 0.0,
                        "tgi": 0.0,
                        "vari": 0.0,
                        "gli": 0.0,
                        "ngrdi": 0.0,
                        "exg": 0.0,
                        "mgrvi": 0.0,
                        "lai_proxy": 0.0,
                        "management_zone": "Zone N/A",
                        "management_action": "No valid crop pixels in this segment.",
                        "recommendation": "No valid crop pixels detected in this segment.",
                        "possible_issue": "No reliable crop signal in this cell (possibly non-crop area or outside selected region).",
                        "empty": True,
                    }
                )

    analysis_payload = {
        "analysis_version": "rgb-proxy-cropped-v1",
        "image_id": image_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "mean_red": round(metrics["mean_r"], 3),
            "mean_green": round(metrics["mean_g"], 3),
            "mean_blue": round(metrics["mean_b"], 3),
            "green_coverage": round(metrics["green_coverage"], 4),
        },
        "rgb_indices": {
            "vari": round(extended_indices["vari"], 4),
            "gli": round(extended_indices["gli"], 4),
            "ngrdi": round(extended_indices["ngrdi"], 4),
            "exg": round(extended_indices["exg"], 4),
            "tgi": round(extended_indices["tgi"], 4),
            "mgrvi": round(extended_indices["mgrvi"], 4),
            "lai_proxy": round(extended_indices["lai_proxy"], 3),
        },
        "farmer_features": {
            "green_coverage_pct": round(metrics["green_coverage"] * 100, 2),
            "estimated_stress_zone_pct": round(metrics["stress_ratio"] * 100, 2),
            "vegetation_vigor_score": round((metrics["avg_vigor"] + 1) * 50, 2),
            "recommended_scan_schedule": "Rescan the selected area in 3-5 days under similar lighting.",
        },
        "vegetation_indices_analysis": {
            "management_zone": zone["zone"],
            "management_action": zone["action"],
            "percent_canopy_cover": round(extended_indices["canopy_cover_pct"], 2),
            "relative_biomass_score": round(extended_indices["relative_biomass_score"], 2),
            "stand_uniformity_score": round(extended_indices["stand_uniformity_score"], 2),
            "relative_yield_potential_pct": round(extended_indices["relative_yield_potential_pct"], 2),
        },
        "report": asdict(report),
    }

    return {
        "crop_analysis": analysis_payload,
        "cropped_original_data_url": _pil_image_to_data_url(masked_original, "PNG"),
        "cropped_heatzone_data_url": _pil_image_to_data_url(heatzone, "PNG"),
        "segments": segments,
        "grid": {"rows": grid_rows, "cols": grid_cols},
    }


def get_result_entry(user_id: str, image_id: str) -> dict[str, Any] | None:
    items = load_user_results_index(user_id)
    for item in items:
        if item.get("image_id") == image_id:
            return item
    return None


def load_result_analysis(entry: dict[str, Any]) -> dict[str, Any] | None:
    inline = entry.get("analysis_json")
    if isinstance(inline, dict):
        return inline

    json_path = entry.get("analysis_json_path")
    if not json_path:
        return None
    p = Path(json_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _try_font(size: int, bold: bool = False):
    try:
        from PIL import ImageFont

        if os.name == "nt":
            font_path = "C:/Windows/Fonts/segoeui.ttf"
            bold_path = "C:/Windows/Fonts/segoeuib.ttf"
            return ImageFont.truetype(bold_path if bold else font_path, size)
        return ImageFont.load_default()
    except Exception:
        try:
            from PIL import ImageFont

            return ImageFont.load_default()
        except Exception:
            return None


def _draw_wrapped(draw, text: str, x: int, y: int, width_chars: int, color, font, line_gap: int = 9) -> int:
    lines = textwrap.wrap(text or "-", width=width_chars) or ["-"]
    line_height = 20
    for line in lines:
        draw.text((x, y), line, fill=color, font=font)
        y += line_height
    return y + line_gap


def _text_width(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(0, bbox[2] - bbox[0])


def _text_height(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(0, bbox[3] - bbox[1])


def _wrap_text_by_pixels(draw, text: str, max_width: int, font) -> list[str]:
    base = (text or "-").strip()
    if not base:
        return ["-"]
    words = base.split()
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines or ["-"]


def _draw_paragraph(draw, text: str, x: int, y: int, width_px: int, color, font, line_gap: int = 8) -> int:
    lines = _wrap_text_by_pixels(draw, text, width_px, font)
    line_height = _text_height(draw, "Ag", font) + 5
    for line in lines:
        draw.text((x, y), line, fill=color, font=font)
        y += line_height
    return y + line_gap


def _draw_card(draw, box: tuple[int, int, int, int], fill: str, outline: str, radius: int = 18, width: int = 2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _fetch_result_image_path(result: dict[str, Any], user_id: str, key: str) -> Path | None:
    storage_key = "original_storage_path" if key == "original" else "heatzone_storage_path"
    bucket = SUPABASE_BUCKET_ORIGINAL if key == "original" else SUPABASE_BUCKET_HEATZONE
    object_path = result.get(storage_key)
    if not object_path:
        return None
    safe_name = f"{result.get('image_id', 'result')}_{key}.png"
    target = ensure_temp_dir(user_id) / safe_name
    if download_file_from_bucket(bucket, object_path, target):
        return target
    return None


def build_result_report_pdf(user_id: str, result: dict[str, Any], analysis: dict[str, Any] | None) -> bytes:
    from PIL import Image, ImageDraw

    width, height = 1240, 1754  # A4-ish @ 150 DPI
    page = Image.new("RGB", (width, height), "#F3F6F8")
    draw = ImageDraw.Draw(page)

    # Modern neutral+green palette for print readability.
    c_header = "#1D4E3D"
    c_header_soft = "#2A6A54"
    c_text_primary = "#18232C"
    c_text_secondary = "#4D5C68"
    c_card = "#FFFFFF"
    c_card_alt = "#F7FAFC"
    c_border = "#DCE4EA"
    c_accent = "#2E7D5B"
    c_badge_bg = "#E8F5EE"

    font_title = _try_font(46, bold=True)
    font_h2 = _try_font(27, bold=True)
    font_h3 = _try_font(22, bold=True)
    font_body = _try_font(20, bold=False)
    font_small = _try_font(17, bold=False)

    personalization = read_result_personalization(analysis)
    report = (analysis or {}).get("report", {}) if isinstance(analysis, dict) else {}
    farmer_features = (analysis or {}).get("farmer_features", {}) if isinstance(analysis, dict) else {}
    title = personalization.get("title") or f"Result {result.get('image_id', '-')}"
    field_name = personalization.get("field_name") or "-"
    crop_type = personalization.get("crop_type") or "-"
    health_band = str(result.get("health_band", "WATCH")).upper()

    status_map = {
        "HEALTHY": ("Good condition", "Plants look generally healthy.", "Keep normal care and continue regular monitoring."),
        "WATCH": ("Needs watching", "Some parts may be under early stress.", "Inspect weaker spots and adjust water/fertilizer early."),
        "RISK": ("At risk", "Visible stress is present in parts of the field.", "Check affected areas today and apply corrective action."),
        "CRITICAL": ("Critical condition", "Crop health is poor in multiple areas.", "Take immediate field action and reassess soon."),
    }
    status_title, status_meaning, status_action = status_map.get(
        health_band,
        ("Needs checking", "The field may have uneven crop health.", "Do a field walk and focus on weak-looking zones."),
    )

    # Header band
    draw.rectangle([0, 0, width, 205], fill=c_header)
    draw.rectangle([0, 150, width, 205], fill=c_header_soft)
    draw.text((64, 36), "AgriVision Report", fill="#FFFFFF", font=font_title)
    draw.text((64, 104), title[:80], fill="#DDF7EA", font=font_h2)

    # Top metadata chips
    meta_y = 158
    chips = [
        f"Result ID: {result.get('image_id', '-')}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Field: {field_name}",
        f"Crop: {crop_type}",
    ]
    chip_x = 64
    for chip in chips:
        chip_w = _text_width(draw, chip, font_small) + 28
        _draw_card(draw, (chip_x, meta_y, chip_x + chip_w, meta_y + 34), c_badge_bg, "#CDE8DA", radius=12, width=1)
        draw.text((chip_x + 14, meta_y + 7), chip, fill=c_header, font=font_small)
        chip_x += chip_w + 12
        if chip_x > width - 260:
            break

    # KPI row
    kpi_top = 228
    kpi_box_h = 94
    kpi_gap = 14
    kpi_w = (width - 64 - 64 - kpi_gap * 2) // 3
    kpis = [
        ("Field Status", status_title),
        ("Health Score (0-100)", str(result.get("health_score", "-"))),
        ("Result Reliability", str(result.get("confidence", "-"))),
    ]
    for idx, (label, value) in enumerate(kpis):
        x1 = 64 + idx * (kpi_w + kpi_gap)
        x2 = x1 + kpi_w
        _draw_card(draw, (x1, kpi_top, x2, kpi_top + kpi_box_h), c_card, c_border, radius=16, width=2)
        draw.text((x1 + 18, kpi_top + 16), label, fill=c_text_secondary, font=font_small)
        draw.text((x1 + 18, kpi_top + 46), value, fill=c_accent, font=font_h3)

    # Image comparison section
    comp_y = 346
    draw.text((64, comp_y), "Image Comparison", fill=c_text_primary, font=font_h2)
    left_panel = (64, comp_y + 42, 602, comp_y + 492)
    right_panel = (638, comp_y + 42, 1176, comp_y + 492)
    _draw_card(draw, left_panel, c_card, c_border, radius=18, width=2)
    _draw_card(draw, right_panel, c_card, c_border, radius=18, width=2)
    draw.text((left_panel[0] + 20, left_panel[1] + 16), "Original Image", fill=c_text_secondary, font=font_small)
    draw.text((right_panel[0] + 20, right_panel[1] + 16), "AI Heatzone", fill=c_text_secondary, font=font_small)

    for key, panel in (("original", left_panel), ("heatzone", right_panel)):
        img_path = _fetch_result_image_path(result, user_id, key)
        inner = (panel[0] + 16, panel[1] + 50, panel[2] - 16, panel[3] - 16)
        if img_path and img_path.exists():
            try:
                with Image.open(img_path) as source:
                    img = source.convert("RGB")
                    box_w = inner[2] - inner[0]
                    box_h = inner[3] - inner[1]
                    img.thumbnail((box_w, box_h))
                    px = inner[0] + (box_w - img.width) // 2
                    py = inner[1] + (box_h - img.height) // 2
                    page.paste(img, (px, py))
            except Exception:
                draw.text((inner[0] + 18, inner[1] + 18), "Image unavailable", fill=c_text_secondary, font=font_body)
        else:
            draw.text((inner[0] + 18, inner[1] + 18), "Image unavailable", fill=c_text_secondary, font=font_body)

    # Summary + metrics row
    body_top = 864
    left_body = (64, body_top, 760, 1508)
    right_body = (786, body_top, 1176, 1508)
    _draw_card(draw, left_body, c_card, c_border, radius=18, width=2)
    _draw_card(draw, right_body, c_card_alt, c_border, radius=18, width=2)

    y = left_body[1] + 24
    draw.text((left_body[0] + 20, y), "Farmer-Friendly Summary", fill=c_text_primary, font=font_h2)
    y += 44
    y = _draw_paragraph(
        draw,
        f"Overall result: {report.get('one_line_summary', status_meaning)}",
        left_body[0] + 20,
        y,
        left_body[2] - left_body[0] - 40,
        c_text_primary,
        font_body,
    )
    y = _draw_paragraph(
        draw,
        f"What this means: {report.get('simple_explanation', status_meaning)}",
        left_body[0] + 20,
        y,
        left_body[2] - left_body[0] - 40,
        c_text_secondary,
        font_body,
    )
    y = _draw_paragraph(
        draw,
        f"What to do now: {status_action}",
        left_body[0] + 20,
        y,
        left_body[2] - left_body[0] - 40,
        c_text_primary,
        font_body,
    )
    y = _draw_paragraph(
        draw,
        f"Field details: Field name {field_name} | Crop type {crop_type}",
        left_body[0] + 20,
        y,
        left_body[2] - left_body[0] - 40,
        c_text_primary,
        font_body,
    )
    y = _draw_paragraph(
        draw,
        f"Farmer Notes: {personalization.get('farmer_notes') or '-'}",
        left_body[0] + 20,
        y,
        left_body[2] - left_body[0] - 40,
        c_text_secondary,
        font_body,
    )

    flags = personalization.get("flags") or []
    if flags:
        y += 4
        draw.text((left_body[0] + 20, y), "Pinned Flags", fill=c_text_primary, font=font_h3)
        y += 38
        for idx, flag in enumerate(flags[:15], start=1):
            y = _draw_paragraph(
                draw,
                f"{idx}. {flag.get('label', 'Marked point')}",
                left_body[0] + 28,
                y,
                left_body[2] - left_body[0] - 56,
                c_text_secondary,
                font_small,
                4,
            )
            if y > left_body[3] - 30:
                break

    # Metrics card content
    metrics_x = right_body[0] + 18
    metrics_y = right_body[1] + 24
    draw.text((metrics_x, metrics_y), "Simple Field Check", fill=c_text_primary, font=font_h2)
    metrics_y += 46
    metric_lines = [
        ("Healthy green area", f"{farmer_features.get('green_coverage_pct', '-')}%"),
        ("Area under stress", f"{farmer_features.get('estimated_stress_zone_pct', '-')}%"),
        ("Plant strength score", f"{farmer_features.get('vegetation_vigor_score', '-')}/100"),
        ("Current status", status_title),
        ("Recommended focus", "Visit weaker-looking sections first"),
        ("Next scan", "Repeat scan after field action"),
    ]
    for label, value in metric_lines:
        _draw_card(
            draw,
            (metrics_x, metrics_y, right_body[2] - 18, metrics_y + 56),
            "#FFFFFF",
            "#D5DEE6",
            radius=12,
            width=1,
        )
        draw.text((metrics_x + 14, metrics_y + 10), label, fill=c_text_secondary, font=font_small)
        draw.text((metrics_x + 14, metrics_y + 30), value, fill=c_text_primary, font=font_small)
        metrics_y += 66

    recommendations = report.get("recommendations", []) if isinstance(report, dict) else []
    rec_box = (64, 1534, 1176, 1688)
    _draw_card(draw, rec_box, c_card, c_border, radius=16, width=2)
    draw.text((84, 1554), "Recommended Actions (Simple Steps)", fill=c_text_primary, font=font_h2)
    if recommendations:
        yy = 1600
        done_states = personalization.get("recommendation_checks") or []
        col_x = [84, 642]
        row_y = [yy, yy + 40, yy + 80]
        for idx, rec in enumerate(recommendations[:6]):
            marker = "[Done]" if idx < len(done_states) and done_states[idx] else "[To do]"
            cx = col_x[idx // 3]
            cy = row_y[idx % 3]
            text = f"{idx + 1}. {marker} {rec}"
            draw.text((cx, cy), text[:95], fill=c_text_secondary, font=font_small)
    else:
        draw.text((84, 1602), "No recommendations available for this result.", fill=c_text_secondary, font=font_small)

    # Footer
    footer = "Generated by AgriVision analytics. Use this report for field guidance and tracking."
    draw.text((64, 1712), footer, fill="#6C7A86", font=font_small)
    right_stamp = datetime.now().strftime("%Y-%m-%d")
    stamp_w = _text_width(draw, right_stamp, font_small)
    draw.text((width - 64 - stamp_w, 1712), right_stamp, fill="#6C7A86", font=font_small)

    output = io.BytesIO()
    try:
        page.save(output, format="PDF", resolution=150.0)
    except TypeError:
        # Pillow version compatibility fallback.
        page.save(output, format="PDF")
    return output.getvalue()


@app.route("/")
def index():
    if not require_auth():
        return redirect(url_for("login_page"))

    user = current_user()
    return render_template(
        "index.html",
        username=to_upper_text(user.get("username", ""), "USER"),
        privilege=to_upper_text(user.get("privilege", ""), "USER"),
        user_api_key=user.get("user_id", ""),
        upload_logs=session.get("upload_logs", []),
        entries=load_user_results_index(user.get("user_id", "")),
    )


@app.route("/results")
def results_page():
    if not require_auth():
        return redirect(url_for("login_page"))
    return redirect(url_for("index", window="results"))


@app.route("/results/<image_id>")
def result_view(image_id: str):
    if not require_auth():
        return redirect(url_for("login_page"))
    user = current_user()
    user_id = user.get("user_id", "")
    entry = get_result_entry(user_id, image_id)
    if not entry:
        return redirect(url_for("results_page"))
    analysis = load_result_analysis(entry)
    personalization = read_result_personalization(analysis)
    return render_template("result-view.html", result=entry, analysis=analysis, personalization=personalization)


@app.get("/results/<image_id>/report.pdf")
def result_report_pdf(image_id: str):
    if not require_auth():
        return redirect(url_for("login_page"))
    user = current_user()
    entry = get_result_entry(user.get("user_id", ""), image_id)
    if not entry:
        return redirect(url_for("results_page"))
    analysis = load_result_analysis(entry)
    pdf_bytes = build_result_report_pdf(user.get("user_id", ""), entry, analysis)
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="result_{image_id}.pdf"'
        },
    )


@app.get("/api/results/<image_id>/customization")
def get_result_customization(image_id: str):
    if not require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    user = current_user()
    entry = get_result_entry(user.get("user_id", ""), image_id)
    if not entry:
        return jsonify({"error": "Result not found"}), 404
    analysis = load_result_analysis(entry)
    personalization = read_result_personalization(analysis)
    return jsonify({"result_id": image_id, "personalization": personalization})


@app.post("/api/results/<image_id>/customization")
def update_result_customization(image_id: str):
    if not require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    user = current_user()
    user_id = user.get("user_id", "")
    entry = get_result_entry(user_id, image_id)
    if not entry:
        return jsonify({"error": "Result not found"}), 404

    payload = request.get_json(silent=True) or {}
    personalization = normalize_result_personalization(payload.get("personalization"))
    ok, err = save_result_personalization(user_id, image_id, personalization)
    if not ok:
        return jsonify({"error": err or "Failed to save customization"}), 500

    return jsonify({"result_id": image_id, "personalization": personalization})


@app.post("/api/results/<image_id>/crop-rerun")
def crop_rerun_result(image_id: str):
    if not require_auth():
        return jsonify({"error": "Unauthorized"}), 401

    user = current_user()
    user_id = user.get("user_id", "")
    entry = get_result_entry(user_id, image_id)
    if not entry:
        return jsonify({"error": "Result not found"}), 404

    payload = request.get_json(silent=True) or {}
    points = _normalize_polygon_points(payload.get("points"))
    if len(points) < 3:
        return jsonify({"error": "Please draw a freeform crop area first."}), 400

    try:
        grid_rows = int(payload.get("grid_rows", 4))
        grid_cols = int(payload.get("grid_cols", 4))
    except (TypeError, ValueError):
        return jsonify({"error": "Grid rows/cols must be valid numbers."}), 400

    grid_rows = min(max(grid_rows, 2), 12)
    grid_cols = min(max(grid_cols, 2), 12)

    original_path, created_temp = _resolve_original_result_image(user_id, image_id, entry)
    if original_path is None or not original_path.exists():
        return jsonify({"error": "Original image unavailable for crop analysis."}), 404

    try:
        response = analyze_freeform_cropped_segments(original_path, image_id, points, grid_rows, grid_cols)
        add_upload_log(user_id, f"Cropped segment analysis completed ({image_id})", "success")
        return jsonify(response)
    except Exception as exc:
        add_upload_log(user_id, f"Cropped segment analysis failed ({image_id}): {exc}", "error")
        return jsonify({"error": str(exc)}), 400
    finally:
        if created_temp:
            try:
                if original_path.exists():
                    original_path.unlink()
            except OSError:
                pass


@app.post("/results/<image_id>/rerun")
def rerun_result(image_id: str):
    if not require_auth():
        return redirect(url_for("login_page"))
    user = current_user()
    user_id = user.get("user_id", "")
    entry = get_result_entry(user_id, image_id)
    if not entry:
        return redirect(url_for("results_page"))

    original_url = entry.get("original_url", "")
    original_path: Path | None = None
    if isinstance(original_url, str) and original_url.startswith("/") and len(original_url.strip()) > 1:
        original_path = BASE_DIR / original_url.lstrip("/")
    created_temp = False

    if original_path is None or not original_path.exists() or not original_path.is_file():
        storage_path = entry.get("original_storage_path")
        if storage_path:
            user_result_img_dir = ensure_temp_dir(str(user_id))
            fallback_name = entry.get("filename") or f"{image_id}.jpg"
            original_path = user_result_img_dir / f"source_{fallback_name}"
            if not download_file_from_bucket(SUPABASE_BUCKET_ORIGINAL, storage_path, original_path):
                add_upload_log(user_id, f"Re-run failed: original image missing ({image_id})", "error")
                return redirect(url_for("result_view", image_id=image_id))
            created_temp = True
        else:
            add_upload_log(user_id, f"Re-run failed: original image missing ({image_id})", "error")
            return redirect(url_for("result_view", image_id=image_id))

    try:
        processed = run_rgb_analysis(original_path, user_id, image_id)
        previous_analysis = load_result_analysis(entry) or {}
        processed["analysis"]["personalization"] = read_result_personalization(previous_analysis)
        entry["updated_at"] = now_utc_iso()
        entry["health_band"] = processed["analysis"]["report"]["model_result"]["health_band"]
        entry["health_score"] = processed["analysis"]["report"]["model_result"]["health_score"]
        entry["confidence"] = processed["analysis"]["report"]["model_result"]["confidence"]
        entry["summary"] = processed["analysis"]["report"]["one_line_summary"]

        heatzone_object_path = f"{user_id}/{image_id}/{processed['heatzone_filename']}"
        local_heatzone_path = Path(processed["heatzone_path"])
        ok_heat, err_heat = upload_file_to_bucket(local_heatzone_path, SUPABASE_BUCKET_HEATZONE, heatzone_object_path)
        require_bucket_upload(
            ok_heat,
            "heatzone image",
            err_heat,
        )
        entry["heatzone_storage_path"] = heatzone_object_path
        entry["heatzone_url"] = signed_or_local_url(SUPABASE_BUCKET_HEATZONE, heatzone_object_path, "")

        report_object_path = f"{user_id}/{image_id}.json"
        ok_json, err_json = upload_json_to_bucket(SUPABASE_BUCKET_REPORTS, report_object_path, processed["analysis"])
        require_bucket_upload(ok_json, "report json", err_json)

        if not upsert_result_in_supabase(
            str(user_id),
            entry,
            processed["analysis"],
        ):
            raise RuntimeError("Supabase table upsert failed.")
        try:
            if local_heatzone_path.exists():
                local_heatzone_path.unlink()
            if created_temp and original_path and original_path.exists():
                original_path.unlink()
        except OSError:
            pass
    except Exception as exc:
        add_upload_log(user_id, f"Analysis re-run failed ({image_id}): {exc}", "error")
        return redirect(url_for("result_view", image_id=image_id))

    add_upload_log(user_id, f"Analysis re-run completed ({image_id})", "success")
    return redirect(url_for("result_view", image_id=image_id))


@app.delete("/api/results/<image_id>")
def delete_result(image_id: str):
    if not require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if supabase is None:
        return jsonify({"error": "Supabase client not initialized."}), 500

    user = current_user()
    user_id = user.get("user_id", "")
    entry = get_result_entry(user_id, image_id)
    if not entry:
        return jsonify({"error": "Result not found"}), 404

    try:
        original_path = entry.get("original_storage_path", "")
        heatzone_path = entry.get("heatzone_storage_path", "")
        report_path = f"{user_id}/{image_id}.json"

        require_bucket_upload(delete_file_from_bucket(SUPABASE_BUCKET_ORIGINAL, original_path), "original image delete")
        require_bucket_upload(delete_file_from_bucket(SUPABASE_BUCKET_HEATZONE, heatzone_path), "heatzone image delete")
        require_bucket_upload(delete_file_from_bucket(SUPABASE_BUCKET_REPORTS, report_path), "report json delete")

        supabase.table(SUPABASE_RESULTS_TABLE).delete().eq("image_id", image_id).execute()
        supabase.table(SUPABASE_IMAGES_TABLE).delete().eq("image_id", image_id).eq("user_id", user_id).execute()
        add_upload_log(user_id, f"Result deleted ({image_id})", "success")
        return jsonify({"ok": True, "result_id": image_id})
    except Exception as exc:
        add_upload_log(user_id, f"Delete failed ({image_id}): {exc}", "error")
        return jsonify({"error": str(exc)}), 500


@app.route("/login", methods=["GET"])
def login_page():
    if require_auth():
        return redirect(url_for("index"))
    success = request.args.get("success", "")
    return render_template("login.html", success=success, error="")


@app.route("/login", methods=["POST"])
def login():
    if supabase is None:
        return render_template(
            "login.html",
            error="Missing Supabase key. Set SUPABASE_KEY (or SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY) in .env.",
            success="",
        )

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("login.html", error="Username and password are required.", success="")

    query = (
        supabase.table(SUPABASE_TABLE)
        .select("user_id, username, password, privilege")
        .eq("username", username)
        .limit(1)
        .execute()
    )

    users = query.data or []
    if not users:
        return render_template("login.html", error="Invalid username or password.", success="")

    db_user = users[0]
    if not verify_password(db_user.get("password", ""), password):
        return render_template("login.html", error="Invalid username or password.", success="")

    session["user"] = {
        "user_id": db_user.get("user_id"),
        "username": db_user.get("username", "USER"),
        "privilege": db_user.get("privilege", "USER"),
    }

    return redirect(url_for("index"))


@app.route("/register", methods=["GET"])
def register_page():
    if require_auth():
        return redirect(url_for("index"))
    return render_template("register.html", error="")


@app.route("/register", methods=["POST"])
def register():
    if supabase is None:
        return render_template(
            "register.html",
            error="Missing Supabase key. Set SUPABASE_KEY (or SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY) in .env.",
        )

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("register.html", error="Username and password are required.")

    existing = (
        supabase.table(SUPABASE_TABLE)
        .select("user_id")
        .eq("username", username)
        .limit(1)
        .execute()
    )

    if existing.data:
        return render_template("register.html", error="Username already exists.")

    payload = {
        "user_id": str(uuid.uuid4()),
        "username": username,
        "password": generate_password_hash(password),
        "privilege": "USER",
    }

    supabase.table(SUPABASE_TABLE).insert(payload).execute()
    return redirect(url_for("login_page", success="Account created. You can now sign in."))


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(_exc):
    if require_auth():
        user = current_user() or {}
        user_id = user.get("user_id", "unknown-user")
        add_upload_log(user_id, "Upload failed: file too large (max 12 MB).", "error")
        return redirect(url_for("index"))
    return redirect(url_for("login_page"))


@app.post("/upload-image")
def upload_image():
    if not require_auth():
        return redirect(url_for("login_page"))

    user = current_user()
    user_id = user.get("user_id", "unknown-user")
    uploaded_file = request.files.get("image")
    file_label = uploaded_file.filename if uploaded_file else "no-file"

    add_upload_log(user_id, f"Upload started ({file_label})", "info")

    if not uploaded_file or not uploaded_file.filename:
        add_upload_log(user_id, "Upload failed: no file selected", "error")
        return redirect(url_for("index"))

    if not is_allowed_image(uploaded_file.filename):
        add_upload_log(user_id, f"Upload failed: unsupported file type ({uploaded_file.filename})", "error")
        return redirect(url_for("index"))

    try:
        safe_name = secure_filename(uploaded_file.filename)
        image_id = str(uuid.uuid4())
        user_upload_dir = ensure_temp_dir(str(user_id))
        stamped_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        original_path = user_upload_dir / stamped_name
        uploaded_file.save(original_path)

        processed = run_rgb_analysis(original_path, str(user_id), image_id)
        processed["analysis"]["personalization"] = default_result_personalization()
        report = processed["analysis"]["report"]
        model_result = report["model_result"]

        entry = {
            "image_id": image_id,
            "filename": stamped_name,
            "created_at": now_utc_iso(),
            "updated_at": now_utc_iso(),
            "original_url": "",
            "heatzone_url": "",
            "analysis_json_path": "",
            "summary": report["one_line_summary"],
            "health_band": model_result["health_band"],
            "health_score": model_result["health_score"],
            "confidence": model_result["confidence"],
        }

        original_object_path = f"{user_id}/{image_id}/{stamped_name}"
        heatzone_object_path = f"{user_id}/{image_id}/{processed['heatzone_filename']}"
        report_object_path = f"{user_id}/{image_id}.json"

        ok_original, err_original = upload_file_to_bucket(original_path, SUPABASE_BUCKET_ORIGINAL, original_object_path)
        require_bucket_upload(ok_original, "original image", err_original)
        entry["original_storage_path"] = original_object_path
        entry["original_url"] = signed_or_local_url(SUPABASE_BUCKET_ORIGINAL, original_object_path, "")

        local_heatzone_path = Path(processed["heatzone_path"])
        ok_heat, err_heat = upload_file_to_bucket(local_heatzone_path, SUPABASE_BUCKET_HEATZONE, heatzone_object_path)
        require_bucket_upload(ok_heat, "heatzone image", err_heat)
        entry["heatzone_storage_path"] = heatzone_object_path
        entry["heatzone_url"] = signed_or_local_url(SUPABASE_BUCKET_HEATZONE, heatzone_object_path, "")

        ok_report, err_report = upload_json_to_bucket(SUPABASE_BUCKET_REPORTS, report_object_path, processed["analysis"])
        require_bucket_upload(ok_report, "report json", err_report)

        persisted_remote = upsert_result_in_supabase(str(user_id), entry, processed["analysis"])
        if not persisted_remote:
            raise RuntimeError("Supabase table upsert failed.")
        try:
            if original_path.exists():
                original_path.unlink()
            if local_heatzone_path.exists():
                local_heatzone_path.unlink()
        except OSError:
            pass
        add_upload_log(user_id, f"Upload succeeded ({stamped_name}) -> Result ID {image_id}", "success")
    except Exception as exc:
        add_upload_log(user_id, f"Upload failed: {exc}", "error")

    return redirect(url_for("index"))


@app.post("/account/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.post("/account/username")
def update_username():
    if not require_auth():
        return jsonify({"error": "Unauthorized"}), 401
    if supabase is None:
        return jsonify({"error": "SUPABASE_KEY not configured"}), 500

    data = request.get_json(silent=True) or {}
    new_username = (data.get("username", "") or "").strip()
    if not new_username:
        return jsonify({"error": "Username is required"}), 400

    user = current_user()
    user_id = user.get("user_id")

    in_use = (
        supabase.table(SUPABASE_TABLE)
        .select("user_id")
        .eq("username", new_username)
        .neq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if in_use.data:
        return jsonify({"error": "Username is already in use"}), 409

    supabase.table(SUPABASE_TABLE).update({"username": new_username}).eq("user_id", user_id).execute()

    user["username"] = new_username
    session["user"] = user
    return jsonify({"username": new_username})


if __name__ == "__main__":
    app.run(debug=True)

