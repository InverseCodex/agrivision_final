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
CROP_SEGMENT_MIN_MASK_COVERAGE = 0.20
CROP_SEGMENT_MIN_VALID_PIXEL_RATIO = 0.35
ORTHO_MAX_EDGE_PX = 1400
ORTHO_MIN_EDGE_PX = 220


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
        "capture_altitude_m": "",
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
    altitude_value = str(data.get("capture_altitude_m", "") or "").strip()
    base["capture_altitude_m"] = altitude_value if altitude_value in {"40", "60", "80", "100"} else ""
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


def load_single_result_entry_from_supabase(user_id: str, image_id: str) -> dict[str, Any] | None:
    if supabase is None:
        return None
    try:
        image_resp = (
            supabase.table(SUPABASE_IMAGES_TABLE)
            .select("image_id, filename, original_path, heatzone_path, uploaded_at, updated_at, status")
            .eq("user_id", user_id)
            .eq("image_id", image_id)
            .limit(1)
            .execute()
        )
        image_rows = image_resp.data or []
        if not image_rows:
            return None

        row = image_rows[0]
        result_resp = (
            supabase.table(SUPABASE_RESULTS_TABLE)
            .select("image_id, analysis_json, summary, health_band, health_score, confidence, created_at, updated_at")
            .eq("image_id", image_id)
            .limit(1)
            .execute()
        )
        result_rows = result_resp.data or []
        res = result_rows[0] if result_rows else {}

        original_ref = row.get("original_path", "")
        heatzone_ref = row.get("heatzone_path", "")
        if not original_ref or not heatzone_ref:
            return None

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
        return entry if entry["original_url"] and entry["heatzone_url"] else None
    except Exception:
        return None


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
        score = _rgb_stress_score(r, g, b)
        critical_cutoff, low_cutoff, watch_cutoff = _stress_band_cutoffs("DJI Mini 4 Pro")

        overlay = np.zeros_like(rgb, dtype=np.uint8)
        mask_critical = score < critical_cutoff
        mask_low = (score >= critical_cutoff) & (score < low_cutoff)
        mask_watch = (score >= low_cutoff) & (score < watch_cutoff)
        mask_good = score >= watch_cutoff

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
        score = _rgb_stress_score(r, g, b)
        critical_cutoff, low_cutoff, watch_cutoff = _stress_band_cutoffs("DJI Mini 4 Pro")

        overlay = np.zeros_like(rgb_np, dtype=np.uint8)
        mask_critical = score < critical_cutoff
        mask_low = (score >= critical_cutoff) & (score < low_cutoff)
        mask_watch = (score >= low_cutoff) & (score < watch_cutoff)
        mask_good = score >= watch_cutoff

        overlay[mask_critical] = (255, 64, 64)
        overlay[mask_low] = (255, 171, 64)
        overlay[mask_watch] = (255, 227, 84)
        overlay[mask_good] = (80, 214, 114)

        blended_np = (0.55 * rgb_np + 0.45 * overlay).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended_np, mode="RGB").save(output_path, format="PNG")

        return float(mask_critical.mean()), float(score.mean())


def run_rgb_analysis(original_path: Path, user_id: str, image_id: str) -> dict[str, Any]:
    camera_model = "DJI Mini 4 Pro"
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
        green_coverage = float(_green_dominant_mask(r, g, b, camera_model).mean())
        dry_coverage = float(_dry_canopy_mask(r, g, b).mean())
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
            green_coverage = float(_green_dominant_mask(r, g, b, camera_model).mean())
            dry_coverage = float(_dry_canopy_mask(r, g, b).mean())

    snapshot = RGBSnapshot(
        mean_red=mean_r,
        mean_green=mean_g,
        mean_blue=mean_b,
        green_coverage=green_coverage,
        dry_coverage=dry_coverage,
        camera_model=camera_model,
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
        "analysis_version": "rgb-proxy-v2-dji-mini4pro",
        "image_id": image_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "mean_red": round(mean_r, 3),
            "mean_green": round(mean_g, 3),
            "mean_blue": round(mean_b, 3),
            "green_coverage": round(green_coverage, 4),
            "dry_coverage": round(dry_coverage, 4),
            "camera_model": camera_model,
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
            "estimated_dry_canopy_pct": round(dry_coverage * 100, 2),
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

    total_pixels = len(pixels)
    masked_count = 0
    valid_count = 0
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    green_pixels = 0
    dry_pixels = 0
    stress_pixels = 0
    vigor_sum = 0.0
    critical_cutoff, _, _ = _stress_band_cutoffs("DJI Mini 4 Pro")

    for (r, g, b), m in zip(pixels, mask_pixels):
        if m <= 0:
            continue
        masked_count += 1
        if _pixel_is_non_crop_deadspace(r, g, b, "DJI Mini 4 Pro"):
            continue
        valid_count += 1
        sum_r += r
        sum_g += g
        sum_b += b
        if _pixel_is_green_dominant(r, g, b, "DJI Mini 4 Pro"):
            green_pixels += 1
        if _pixel_is_dry_canopy(r, g, b):
            dry_pixels += 1
        score = _pixel_stress_score(r, g, b)
        vigor_sum += score
        if score < critical_cutoff:
            stress_pixels += 1

    if valid_count <= 0:
        raise RuntimeError("Crop area is empty. Draw a larger crop region.")

    mean_r = sum_r / valid_count
    mean_g = sum_g / valid_count
    mean_b = sum_b / valid_count
    green_coverage = green_pixels / valid_count
    dry_coverage = dry_pixels / valid_count
    stress_ratio = stress_pixels / valid_count
    avg_vigor = vigor_sum / valid_count

    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "green_coverage": green_coverage,
        "dry_coverage": dry_coverage,
        "stress_ratio": stress_ratio,
        "avg_vigor": avg_vigor,
        "valid_count": float(valid_count),
        "mask_count": float(masked_count),
        "mask_coverage": (masked_count / total_pixels) if total_pixels > 0 else 0.0,
        "valid_pixel_ratio": (valid_count / masked_count) if masked_count > 0 else 0.0,
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


def _stress_band_cutoffs(camera_model: str) -> tuple[float, float, float]:
    if "mini 4 pro" in camera_model.strip().lower():
        return (-0.01, 0.05, 0.12)
    return (0.02, 0.08, 0.15)


def _rgb_stress_score(r: Any, g: Any, b: Any) -> Any:
    return (g - r) / (r + g + b + 1e-6)


def _pixel_stress_score(r: float, g: float, b: float) -> float:
    return float((g - r) / (r + g + b + 1e-6))


def _green_dominant_mask(r: Any, g: Any, b: Any, camera_model: str) -> Any:
    ratio = 1.02 if "mini 4 pro" in camera_model.strip().lower() else 1.05
    return (g > r * ratio) & (g > b * ratio)


def _pixel_is_green_dominant(r: int, g: int, b: int, camera_model: str) -> bool:
    ratio = 1.02 if "mini 4 pro" in camera_model.strip().lower() else 1.05
    return g > r * ratio and g > b * ratio


def _dry_canopy_mask(r: Any, g: Any, b: Any) -> Any:
    return (r > b * 1.08) & (g > b * 1.02) & (np.abs(r - g) <= 48)


def _pixel_is_dry_canopy(r: int, g: int, b: int) -> bool:
    return r > b * 1.08 and g > b * 1.02 and abs(r - g) <= 48


def _pixel_is_non_crop_deadspace(r: int, g: int, b: int, camera_model: str = "DJI Mini 4 Pro") -> bool:
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    chroma = max_c - min_c
    near_white = max_c >= 245 and chroma <= 8
    near_black = max_c <= 12 and chroma <= 8
    if near_white or near_black:
        return True

    score = _pixel_stress_score(r, g, b)
    green_dominant = _pixel_is_green_dominant(r, g, b, camera_model)

    muted_surface = chroma <= 30 and 95 <= max_c <= 225 and score < 0.08 and not green_dominant
    tan_padding = chroma <= 44 and max_c >= 125 and abs(r - g) <= 28 and b <= g + 26 and score < 0.10
    dusty_pink = chroma <= 42 and max_c >= 120 and abs(r - b) <= 22 and g <= r + 10 and score < 0.06
    dull_brown = (
        chroma <= 52
        and max_c >= 110
        and r >= g >= b
        and (r - g) <= 34
        and (g - b) <= 26
        and score < 0.10
        and not green_dominant
    )
    return muted_surface or tan_padding or dusty_pink or dull_brown


def _build_effective_crop_mask(rgb_image: Any, mask_image: Any | None = None) -> Any:
    from collections import deque
    from PIL import Image

    rgb = np.asarray(rgb_image.convert("RGB"), dtype=np.int16)
    if mask_image is not None:
        base_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    else:
        base_mask = np.ones(rgb.shape[:2], dtype=bool)

    if not np.any(base_mask):
        return Image.new("L", (rgb.shape[1], rgb.shape[0]), 0)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    chroma = max_c - min_c
    score = (g - r) / (r + g + b + 1e-6)
    green_dominant = (g > r * 1.02) & (g > b * 1.02)

    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    texture = np.zeros_like(gray, dtype=np.float32)
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])
    texture[:, 1:] += diff_x
    texture[:, :-1] += diff_x
    texture[1:, :] += diff_y
    texture[:-1, :] += diff_y
    texture /= 4.0

    near_white = (max_c >= 245) & (chroma <= 8)
    near_black = (max_c <= 12) & (chroma <= 8)
    muted_surface = (chroma <= 30) & (max_c >= 95) & (max_c <= 225) & (score < 0.08) & (~green_dominant)
    tan_padding = (chroma <= 44) & (max_c >= 125) & (np.abs(r - g) <= 28) & (b <= g + 26) & (score < 0.10)
    dusty_pink = (chroma <= 42) & (max_c >= 120) & (np.abs(r - b) <= 22) & (g <= r + 10) & (score < 0.06)
    dull_brown = (
        (chroma <= 52)
        & (max_c >= 110)
        & (r >= g)
        & (g >= b)
        & ((r - g) <= 34)
        & ((g - b) <= 26)
        & (score < 0.10)
        & (~green_dominant)
    )
    flat_muted_deadspace = (
        (chroma <= 58)
        & (max_c >= 90)
        & (max_c <= 235)
        & (texture <= 9.0)
        & (~green_dominant)
    )

    hard_non_crop = near_white | near_black
    candidate_deadspace = muted_surface | tan_padding | dusty_pink | dull_brown | flat_muted_deadspace
    border_deadspace = np.zeros_like(base_mask, dtype=bool)
    height, width = base_mask.shape
    queue: deque[tuple[int, int]] = deque()

    def enqueue_if_candidate(y: int, x: int) -> None:
        if not base_mask[y, x] or not candidate_deadspace[y, x] or border_deadspace[y, x]:
            return
        border_deadspace[y, x] = True
        queue.append((y, x))

    for x in range(width):
        enqueue_if_candidate(0, x)
        enqueue_if_candidate(height - 1, x)
    for y in range(height):
        enqueue_if_candidate(y, 0)
        enqueue_if_candidate(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            enqueue_if_candidate(y - 1, x)
        if y + 1 < height:
            enqueue_if_candidate(y + 1, x)
        if x > 0:
            enqueue_if_candidate(y, x - 1)
        if x + 1 < width:
            enqueue_if_candidate(y, x + 1)

    non_crop = hard_non_crop | border_deadspace
    effective_mask = base_mask & (~non_crop)

    return Image.fromarray((effective_mask.astype(np.uint8) * 255), mode="L")


def _management_zone_recommendation(indices: dict[str, float], stress_ratio: float) -> dict[str, str]:
    tgi = float(indices.get("tgi", 0.0))
    vari = float(indices.get("vari", 0.0))
    canopy = float(indices.get("canopy_cover_pct", 0.0))
    yield_potential = float(indices.get("relative_yield_potential_pct", 0.0))

    if tgi < -0.03 or vari < 0.03 or stress_ratio > 0.42:
        zone = "Zone A - Priority Nitrogen & Water"
        action = "Investigate irrigation distribution and apply targeted nitrogen in weak blocks."
    elif canopy < 40.0 or yield_potential < 42.0:
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

    if band in {"critical", "stressed"} or stress >= 42:
        return "High-risk segment: prioritize field visit, nitrogen correction, and irrigation check."
    if band == "mature":
        return "Segment looks mature or close to harvest. Confirm crop stage and prioritize harvest-readiness checks."
    if canopy < 40 or tgi < -0.03:
        return "Low canopy/chlorophyll signal: monitor for thin stands and apply corrective feeding as needed."
    return "Segment is stable. Maintain current management and continue periodic scans."


def _segment_possible_issue(segment: dict[str, Any]) -> str:
    if segment.get("empty"):
        return "No reliable crop signal in this cell (possibly non-crop area or outside selected region)."

    band = str(segment.get("health_band", "watch")).lower()
    stress = float(segment.get("estimated_stress_zone_pct", 0.0))
    canopy = float(segment.get("canopy_cover_pct", 0.0))
    tgi = float(segment.get("tgi", 0.0))

    if band in {"critical", "stressed"} or stress >= 42:
        return "Possible water stress or uneven irrigation, with potential nutrient deficiency in this section."
    if band == "mature":
        return "Canopy appears dry and mature in this section, which may reflect harvest stage rather than active stress."
    if canopy < 40:
        return "Possible sparse stand or early growth delay compared to surrounding segments."
    if tgi < -0.03:
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
        if _pixel_is_non_crop_deadspace(r, g, b, "DJI Mini 4 Pro"):
            rendered.append((0, 0, 0, 0))
            continue

        score = _pixel_stress_score(r, g, b)
        critical_cutoff, low_cutoff, watch_cutoff = _stress_band_cutoffs("DJI Mini 4 Pro")
        if score < critical_cutoff:
            overlay = (255, 64, 64)
        elif score < low_cutoff:
            overlay = (255, 171, 64)
        elif score < watch_cutoff:
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


def _derive_adaptive_grid_dimensions(
    target_cells: int,
    ref_width: int,
    ref_height: int,
    min_dim: int = 2,
    max_dim: int = 12,
) -> tuple[int, int]:
    target = max(min_dim * min_dim, min(max_dim * max_dim, int(target_cells)))
    safe_w = max(int(ref_width), 1)
    safe_h = max(int(ref_height), 1)
    aspect = safe_w / safe_h

    cols = int(round((target * aspect) ** 0.5))
    cols = max(min_dim, min(max_dim, cols))
    rows = int(round(target / max(cols, 1)))
    rows = max(min_dim, min(max_dim, rows))

    # Keep total cells near target while preserving area aspect.
    while rows * cols < target:
        if cols < max_dim and (cols / max(rows, 1)) < aspect:
            cols += 1
        elif rows < max_dim:
            rows += 1
        else:
            break

    while rows * cols > target + max(cols, rows):
        if rows > min_dim and (rows / max(cols, 1)) > (1.0 / max(aspect, 1e-6)):
            rows -= 1
        elif cols > min_dim:
            cols -= 1
        else:
            break

    return rows, cols


def _derive_mask_weighted_edges(mask_image: Any, parts: int, axis: str) -> list[int]:
    mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    length = mask.shape[1] if axis == "x" else mask.shape[0]
    if parts <= 1 or length <= 1:
        return [0, length]

    occupancy = mask.sum(axis=0 if axis == "x" else 1).astype(np.float64)
    if occupancy.sum() <= 0:
        step = length / parts
        edges = [0]
        for idx in range(1, parts):
            edges.append(min(length - 1, max(edges[-1] + 1, int(round(idx * step)))))
        edges.append(length)
        return edges

    cumulative = np.cumsum(occupancy)
    total = cumulative[-1]
    edges = [0]
    for idx in range(1, parts):
        target = total * (idx / parts)
        edge = int(np.searchsorted(cumulative, target, side="left")) + 1
        edge = min(length - 1, max(edges[-1] + 1, edge))
        edges.append(edge)
    edges.append(length)
    return edges


def _order_quad_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # Returns points in TL, TR, BR, BL order.
    if len(points) != 4:
        return points
    pts = [(float(x), float(y)) for x, y in points]
    sums = [x + y for x, y in pts]
    diffs = [x - y for x, y in pts]
    tl = pts[sums.index(min(sums))]
    br = pts[sums.index(max(sums))]
    tr = pts[diffs.index(max(diffs))]
    bl = pts[diffs.index(min(diffs))]
    return [
        (int(round(tl[0])), int(round(tl[1]))),
        (int(round(tr[0])), int(round(tr[1]))),
        (int(round(br[0])), int(round(br[1]))),
        (int(round(bl[0])), int(round(bl[1]))),
    ]


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5


def _estimate_rectified_size(quad_points: list[tuple[int, int]]) -> tuple[int, int]:
    if len(quad_points) != 4:
        return ORTHO_MIN_EDGE_PX, ORTHO_MIN_EDGE_PX
    tl, tr, br, bl = quad_points
    top_w = _distance(tl, tr)
    bottom_w = _distance(bl, br)
    left_h = _distance(tl, bl)
    right_h = _distance(tr, br)
    est_w = int(round((top_w + bottom_w) * 0.5))
    est_h = int(round((left_h + right_h) * 0.5))
    out_w = max(ORTHO_MIN_EDGE_PX, min(ORTHO_MAX_EDGE_PX, est_w))
    out_h = max(ORTHO_MIN_EDGE_PX, min(ORTHO_MAX_EDGE_PX, est_h))
    return out_w, out_h


def _enhance_for_model(rgb_image: Any) -> Any:
    from PIL import ImageEnhance, ImageFilter

    enhanced = ImageEnhance.Contrast(rgb_image).enhance(1.08)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.04)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.1, percent=90, threshold=2))
    return enhanced


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
        if max(rgb.size) > ANALYSIS_MAX_DIMENSION:
            rgb.thumbnail((ANALYSIS_MAX_DIMENSION, ANALYSIS_MAX_DIMENSION), Image.Resampling.LANCZOS)

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

    analysis_rgb = rgb
    analysis_mask = polygon_mask
    preprocessing_stage = "masked-selection"

    # Lightweight orthorectification (not multi-image stitching) for 4-corner selections.
    if len(px_points) == 4:
        ordered_quad = _order_quad_points(px_points)
        out_w, out_h = _estimate_rectified_size(ordered_quad)
        quad_data = tuple(float(v) for xy in ordered_quad for v in xy)
        rectified = rgb.transform(
            (out_w, out_h),
            Image.Transform.QUAD,
            quad_data,
            resample=Image.Resampling.BICUBIC,
        )
        analysis_rgb = _enhance_for_model(rectified.convert("RGB"))
        analysis_mask = Image.new("L", (out_w, out_h), 255)
        preprocessing_stage = "orthorectified-selection"

    analysis_mask = _build_effective_crop_mask(analysis_rgb, analysis_mask)

    bbox = analysis_mask.getbbox()
    if not bbox:
        raise RuntimeError("Invalid crop selection. Please draw a larger area.")

    analysis_rgb = analysis_rgb.crop(bbox)
    analysis_mask = analysis_mask.crop(bbox)
    bbox_w = max(1, analysis_rgb.size[0])
    bbox_h = max(1, analysis_rgb.size[1])

    masked_original = analysis_rgb.convert("RGBA")
    masked_original.putalpha(analysis_mask)

    metrics = _masked_rgb_metrics(analysis_rgb, analysis_mask)
    snapshot = RGBSnapshot(
        mean_red=metrics["mean_r"],
        mean_green=metrics["mean_g"],
        mean_blue=metrics["mean_b"],
        green_coverage=metrics["green_coverage"],
        dry_coverage=metrics["dry_coverage"],
        camera_model="DJI Mini 4 Pro",
    )
    context = FieldContext(crop_name="Farm Crop", growth_stage="unknown", rainfall_last_7d_mm=0.0, avg_temp_c=0.0)
    report = interpret_field_from_rgb(snapshot, context)
    extended_indices = _compute_extended_rgb_indices(
        metrics["mean_r"], metrics["mean_g"], metrics["mean_b"], metrics["green_coverage"]
    )
    zone = _management_zone_recommendation(extended_indices, stress_ratio=metrics["stress_ratio"])

    heatzone = _build_masked_heatzone_image(analysis_rgb, analysis_mask)

    requested_cells = max(4, min(144, int(grid_rows) * int(grid_cols)))
    adaptive_rows, adaptive_cols = _derive_adaptive_grid_dimensions(requested_cells, bbox_w, bbox_h)

    analysis_width, analysis_height = analysis_rgb.size
    col_edges = _derive_mask_weighted_edges(analysis_mask, adaptive_cols, "x")
    row_edges = _derive_mask_weighted_edges(analysis_mask, adaptive_rows, "y")
    segments: list[dict[str, Any]] = []

    segment_index = 1
    for row in range(adaptive_rows):
        for col in range(adaptive_cols):
            x0 = col_edges[col]
            x1 = col_edges[col + 1]
            y0 = row_edges[row]
            y1 = row_edges[row + 1]
            if x1 <= x0 or y1 <= y0:
                continue

            region_rgb = analysis_rgb.crop((x0, y0, x1, y1))
            region_mask = analysis_mask.crop((x0, y0, x1, y1))
            try:
                region_metrics = _masked_rgb_metrics(region_rgb, region_mask)
                if region_metrics["mask_coverage"] < CROP_SEGMENT_MIN_MASK_COVERAGE:
                    raise RuntimeError("Segment outside selected crop area.")
                if region_metrics["valid_pixel_ratio"] < CROP_SEGMENT_MIN_VALID_PIXEL_RATIO:
                    raise RuntimeError("Segment has too many blank/padding pixels.")
                region_snapshot = RGBSnapshot(
                    mean_red=region_metrics["mean_r"],
                    mean_green=region_metrics["mean_g"],
                    mean_blue=region_metrics["mean_b"],
                    green_coverage=region_metrics["green_coverage"],
                    dry_coverage=region_metrics["dry_coverage"],
                    camera_model="DJI Mini 4 Pro",
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
                    "segment_id": str(segment_index),
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
                    "mask_coverage_pct": round(region_metrics["mask_coverage"] * 100, 2),
                    "valid_pixel_ratio_pct": round(region_metrics["valid_pixel_ratio"] * 100, 2),
                    "management_zone": region_zone["zone"],
                    "management_action": region_zone["action"],
                }
                segment_payload["recommendation"] = _segment_recommendation(segment_payload)
                segment_payload["possible_issue"] = _segment_possible_issue(segment_payload)
                segments.append(segment_payload)
            except RuntimeError:
                segments.append(
                    {
                        "segment_id": str(segment_index),
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
                        "mask_coverage_pct": 0.0,
                        "valid_pixel_ratio_pct": 0.0,
                        "management_zone": "Zone N/A",
                        "management_action": "No valid crop pixels in this segment.",
                        "recommendation": "No valid crop pixels detected in this segment.",
                        "possible_issue": "No reliable crop signal in this cell (possibly non-crop area or outside selected region).",
                        "empty": True,
                    }
                )
            segment_index += 1

    analysis_payload = {
        "analysis_version": "rgb-proxy-masked-selection-v4-dji-mini4pro",
        "image_id": image_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing": {
            "stage": preprocessing_stage,
            "max_dimension_px": ANALYSIS_MAX_DIMENSION,
            "orthorectify_target_max_edge_px": ORTHO_MAX_EDGE_PX,
        },
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
        "grid": {
            "rows": adaptive_rows,
            "cols": adaptive_cols,
            "row_weights": [max(1, row_edges[idx + 1] - row_edges[idx]) for idx in range(adaptive_rows)],
            "col_weights": [max(1, col_edges[idx + 1] - col_edges[idx]) for idx in range(adaptive_cols)],
        },
    }


def get_result_entry(user_id: str, image_id: str) -> dict[str, Any] | None:
    items = load_user_results_index(user_id)
    for item in items:
        if str(item.get("image_id")) == str(image_id):
            return item
    return load_single_result_entry_from_supabase(user_id, image_id)


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


def _pdf_theme_palette(theme: str) -> dict[str, str]:
    safe_theme = (theme or "default").strip().lower()
    palettes = {
        "default": {
            "page": "#F3F6F8",
            "header": "#1D4E3D",
            "header_soft": "#2A6A54",
            "text_primary": "#18232C",
            "text_secondary": "#4D5C68",
            "card": "#FFFFFF",
            "card_alt": "#F7FAFC",
            "border": "#DCE4EA",
            "accent": "#2E7D5B",
            "badge_bg": "#E8F5EE",
            "header_text": "#FFFFFF",
            "header_subtext": "#DDF7EA",
            "footer": "#6C7A86",
        },
        "neon": {
            "page": "#F7F3FF",
            "header": "#220B47",
            "header_soft": "#3B1D73",
            "text_primary": "#1F1436",
            "text_secondary": "#5E4F82",
            "card": "#FFFFFF",
            "card_alt": "#F5EEFF",
            "border": "#D8C8FF",
            "accent": "#00A7B7",
            "badge_bg": "#EFE5FF",
            "header_text": "#FFFFFF",
            "header_subtext": "#E9DDFF",
            "footer": "#7A6B98",
        },
        "minimalist": {
            "page": "#F2F4F5",
            "header": "#2A6F97",
            "header_soft": "#5B8FB1",
            "text_primary": "#182126",
            "text_secondary": "#4B5963",
            "card": "#FFFFFF",
            "card_alt": "#F8FAFB",
            "border": "#D7E0E6",
            "accent": "#215A7A",
            "badge_bg": "#E7F0F6",
            "header_text": "#FFFFFF",
            "header_subtext": "#E6F2F9",
            "footer": "#6D7B86",
        },
    }
    return palettes.get(safe_theme, palettes["default"])


def build_result_report_pdf(user_id: str, result: dict[str, Any], analysis: dict[str, Any] | None, theme: str = "default") -> bytes:
    from PIL import Image, ImageDraw

    width, height = 1240, 1754  # A4-ish @ 150 DPI
    palette = _pdf_theme_palette(theme)
    page = Image.new("RGB", (width, height), palette["page"])
    draw = ImageDraw.Draw(page)

    c_header = palette["header"]
    c_header_soft = palette["header_soft"]
    c_text_primary = palette["text_primary"]
    c_text_secondary = palette["text_secondary"]
    c_card = palette["card"]
    c_card_alt = palette["card_alt"]
    c_border = palette["border"]
    c_accent = palette["accent"]
    c_badge_bg = palette["badge_bg"]

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
        "MATURE": ("Mature stage", "The field appears close to harvest or late-season dry-down.", "Confirm crop stage and plan harvest or final field checks."),
        "WATCH": ("Needs watching", "Some parts may be under early stress.", "Inspect weaker spots and adjust water/fertilizer early."),
        "STRESSED": ("At risk", "Visible stress is present in parts of the field.", "Check affected areas today and apply corrective action."),
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
    draw.text((64, 36), "AgriVision Report", fill=palette["header_text"], font=font_title)
    draw.text((64, 104), title[:80], fill=palette["header_subtext"], font=font_h2)

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
            c_card,
            c_border,
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
    draw.text((64, 1712), footer, fill=palette["footer"], font=font_small)
    right_stamp = datetime.now().strftime("%Y-%m-%d")
    stamp_w = _text_width(draw, right_stamp, font_small)
    draw.text((width - 64 - stamp_w, 1712), right_stamp, fill=palette["footer"], font=font_small)

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


@app.route("/tutorial")
def tutorial_page():
    if not require_auth():
        return redirect(url_for("login_page"))

    user = current_user()
    return render_template(
        "tutorial.html",
        username=to_upper_text(user.get("username", ""), "USER"),
        privilege=to_upper_text(user.get("privilege", ""), "USER"),
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
    pdf_theme = request.args.get("theme", "default")
    pdf_bytes = build_result_report_pdf(user.get("user_id", ""), entry, analysis, theme=pdf_theme)
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
        return redirect(url_for("result_view", image_id=image_id))
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

