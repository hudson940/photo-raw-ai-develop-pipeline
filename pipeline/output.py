"""Stages 7-8: Output orchestration, archive, decision log, and review gate.

Consumes rows in the developed/retouched state and:
- Copies the final image to output/ with an organised name
- Moves the RAW to archive/ and writes a RapidRaw `.rrdata` sidecar of the develop params next to it
- Writes a decision log (JSON) recording every pipeline decision
- Handles photos in the review state via a simple CLI gate
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageOps

from .config import CONFIG
from .storage import STORAGE

log = logging.getLogger("output")


def _to_jpeg(src: Path, dest: Path, quality: int) -> None:
    """Convert a developed/retouched image (16-bit TIFF, PNG, JPEG) to an 8-bit JPEG."""
    if src.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(src))
        if arr.dtype == np.uint16:
            arr = (arr.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = ImageOps.exif_transpose(Image.open(src))
    img.convert("RGB").save(dest, "JPEG", quality=quality, subsampling="4:2:0", optimize=True)


def cleanup_intermediates(stem: str) -> None:
    """Delete the scratch TIFFs (developed + retouched) for a finished photo."""
    if CONFIG.keep_intermediate_tiffs:
        return
    for pattern in (f"{stem}.tiff", f"{stem}.retouched.tiff"):
        p = CONFIG.work / pattern
        if p.exists():
            p.unlink()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _decision_log(photo_id: int, filename: str, state: str, analysis_json: str | None) -> dict:
    ts = _timestamp()
    log_entry = {
        "photo_id": photo_id,
        "filename": filename,
        "final_state": state,
        "exported_at": ts,
    }
    if analysis_json:
        log_entry["analysis"] = json.loads(analysis_json)
    return log_entry


def publish_done(photo_id: int, filename: str, image_path: Path) -> Path:
    """Publish the final image to output/ as a JPEG named <originalname>_<db id>.jpg.

    Using the queue id (not a timestamp) keeps one stable output per photo, so
    re-rendering the same photo overwrites its file instead of piling up copies.
    """
    stem = Path(filename).stem
    out_name = f"{stem}_{photo_id}.jpg"
    out_path = CONFIG.output / out_name
    _to_jpeg(image_path, out_path, CONFIG.output_jpeg_quality)
    size_mb = out_path.stat().st_size / 1e6
    log.info("Published %s -> %s (%.1f MB, q%d)", image_path.name, out_name, size_mb,
             CONFIG.output_jpeg_quality)
    STORAGE.put(out_path)
    return out_path


def archive_raw(raw_path: Path) -> Path:
    """Move a processed RAW to the archive directory (and mirror it to object storage)."""
    dest = CONFIG.archive / raw_path.name
    if raw_path.exists():
        shutil.move(str(raw_path), str(dest))
        log.info("Archived %s", raw_path.name)
        STORAGE.put(dest)
    return dest


# RapidRaw (.rrdata) stores a JSON sidecar with a full `adjustments` object. This is the
# complete default set for RapidRaw's schema; we deep-copy it and override the develop fields
# we can map. Full object (not partial) so RapidRaw's deserializer gets every field it expects.
_RAPIDRAW_DEFAULT_ADJUSTMENTS = r'''{"aiPatches":[],"aspectRatio":null,"blacks":0,"brightness":0,"centré":0,"chromaticAberrationBlueYellow":0,"chromaticAberrationRedCyan":0,"clarity":0,"colorCalibration":{"blueHue":0,"blueSaturation":0,"greenHue":0,"greenSaturation":0,"redHue":0,"redSaturation":0,"shadowsTint":0},"colorGrading":{"balance":0,"blending":50,"global":{"hue":0,"luminance":0,"saturation":0},"highlights":{"hue":0,"luminance":0,"saturation":0},"midtones":{"hue":0,"luminance":0,"saturation":0},"shadows":{"hue":0,"luminance":0,"saturation":0}},"colorNoiseReduction":0,"contrast":0,"crop":null,"curveMode":"point","curves":{"blue":[{"x":0,"y":0},{"x":255,"y":255}],"green":[{"x":0,"y":0},{"x":255,"y":255}],"luma":[{"x":0,"y":0},{"x":255,"y":255}],"red":[{"x":0,"y":0},{"x":255,"y":255}]},"dehaze":0,"exposure":0,"flareAmount":0,"flipHorizontal":false,"flipVertical":false,"glowAmount":0,"grainAmount":0,"grainRoughness":50,"grainSize":25,"halationAmount":0,"highlights":0,"hsl":{"aquas":{"hue":0,"luminance":0,"saturation":0},"blues":{"hue":0,"luminance":0,"saturation":0},"greens":{"hue":0,"luminance":0,"saturation":0},"magentas":{"hue":0,"luminance":0,"saturation":0},"oranges":{"hue":0,"luminance":0,"saturation":0},"purples":{"hue":0,"luminance":0,"saturation":0},"reds":{"hue":0,"luminance":0,"saturation":0},"yellows":{"hue":0,"luminance":0,"saturation":0}},"hue":0,"lensCorrectionMode":"manual","lensDistortionAmount":100,"lensDistortionEnabled":true,"lensMaker":null,"lensModel":null,"lensTcaAmount":100,"lensTcaEnabled":true,"lensVignetteAmount":100,"lensVignetteEnabled":true,"lumaNoiseReduction":0,"lutData":null,"lutIntensity":100,"lutName":null,"lutPath":null,"lutSize":0,"masks":[],"orientationSteps":0,"parametricCurve":{"blue":{"blackLevel":0,"darks":0,"highlights":0,"lights":0,"shadows":0,"split1":25,"split2":50,"split3":75,"whiteLevel":0},"green":{"blackLevel":0,"darks":0,"highlights":0,"lights":0,"shadows":0,"split1":25,"split2":50,"split3":75,"whiteLevel":0},"luma":{"blackLevel":0,"darks":0,"highlights":0,"lights":0,"shadows":0,"split1":25,"split2":50,"split3":75,"whiteLevel":0},"red":{"blackLevel":0,"darks":0,"highlights":0,"lights":0,"shadows":0,"split1":25,"split2":50,"split3":75,"whiteLevel":0}},"pointCurves":{"blue":[{"x":0,"y":0},{"x":255,"y":255}],"green":[{"x":0,"y":0},{"x":255,"y":255}],"luma":[{"x":0,"y":0},{"x":255,"y":255}],"red":[{"x":0,"y":0},{"x":255,"y":255}]},"rotation":0,"saturation":0,"sectionVisibility":{"basic":true,"color":true,"curves":true,"details":true,"effects":true},"shadows":0,"sharpness":0,"sharpnessThreshold":15,"showClipping":false,"structure":0,"temperature":0,"tint":0,"toneMapper":"basic","transformAspect":0,"transformDistortion":0,"transformHorizontal":0,"transformRotate":0,"transformScale":100,"transformVertical":0,"transformXOffset":0,"transformYOffset":0,"vibrance":0,"vignetteAmount":0,"vignetteFeather":50,"vignetteMidpoint":50,"vignetteRoundness":0,"whites":0}'''


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def write_rapidraw_sidecar(filename: str, analysis_json: str | None) -> Path | None:
    """Write a RapidRaw `.rrdata` sidecar next to the archived RAW, mapping our develop
    parameters onto RapidRaw's adjustment sliders so the photo opens in RapidRaw already
    developed. Returns the sidecar path, or None if there's nothing to record.

    Only the develop (tonal) parameters transfer — RapidRaw is a RAW developer, so the
    face/skin/hair/background retouch has no equivalent there.
    """
    if not analysis_json:
        return None
    dp = json.loads(analysis_json).get("develop", {})
    adj = json.loads(_RAPIDRAW_DEFAULT_ADJUSTMENTS)

    adj["exposure"] = round(_clamp(float(dp.get("exposure_ev", 0)), -5, 5), 4)        # EV, -5..5
    adj["contrast"] = round(_clamp(float(dp.get("contrast", 0)) * 100, -100, 100), 2)  # -1..1 -> -100..100
    adj["highlights"] = round(_clamp(float(dp.get("highlights", 0)), -100, 100), 2)
    adj["shadows"] = round(_clamp(float(dp.get("shadows", 0)), -100, 100), 2)
    adj["saturation"] = round(_clamp(float(dp.get("saturation", 0)) * 100, -100, 100), 2)
    adj["vibrance"] = round(_clamp(float(dp.get("vibrance", 0)) * 100, -100, 100), 2)
    # White balance: leave RapidRaw at as-shot (temperature/tint = 0). RapidRaw's temperature
    # is a relative, non-Kelvin slider we can't calibrate 1:1, and our pipeline develops from
    # the camera's as-shot WB anyway — so as-shot gives RapidRaw the correct, cast-free WB
    # (mapping the Kelvin value onto the slider produced a wrong cast).
    adj["temperature"] = 0
    adj["tint"] = 0
    adj["rotation"] = round(float(dp.get("rotation_deg", 0)), 3)

    payload = {"version": 1, "rating": 0, "adjustments": adj, "tags": None, "exif": None}
    dest = CONFIG.archive / f"{filename}.rrdata"
    if dest.exists():                       # preserve RapidRaw's own metadata cache if present
        try:
            old = json.loads(dest.read_text())
            for k in ("version", "rating", "tags", "exif"):
                if k in old:
                    payload[k] = old[k]
        except (ValueError, OSError):
            pass
    CONFIG.archive.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    log.info("Wrote RapidRaw sidecar %s", dest.name)
    STORAGE.put(dest)
    return dest


def write_decision_log(log_entry: dict) -> Path:
    """Append a decision entry to the pipeline decision log."""
    log_path = CONFIG.root / "decision_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    STORAGE.put(log_path)
    return log_path


def approve_review(photo_id: int, conn) -> bool:
    """Move a review-state photo back to analyzed so it proceeds through remaining stages."""
    import sqlite3
    from . import db as dbmod

    row = conn.execute("SELECT * FROM photos WHERE id = ? AND state = 'review'", (photo_id,)).fetchone()
    if row is None:
        return False

    dbmod.set_state(conn, photo_id, "analyzed")
    log.info("Review #%d approved by human — proceeding to develop", photo_id)

    log_entry = _decision_log(photo_id, row["filename"], "review_approved", row["analysis_json"])
    log_entry["decision"] = "human_approved"
    write_decision_log(log_entry)
    return True


def reject_review(photo_id: int, conn, reason: str = "human rejected") -> bool:
    """Mark a review-state photo as failed (human rejected)."""
    from . import db as dbmod

    row = conn.execute("SELECT * FROM photos WHERE id = ? AND state = 'review'", (photo_id,)).fetchone()
    if row is None:
        return False

    dbmod.set_state(conn, photo_id, "failed", error=reason)
    log.info("Review #%d rejected: %s", photo_id, reason)

    log_entry = _decision_log(photo_id, row["filename"], "review_rejected", row["analysis_json"])
    log_entry["decision"] = reason
    write_decision_log(log_entry)
    return True


def finalize(
    photo_id: int,
    filename: str,
    raw_path: Path,
    image_path: Path,
    state: str,
    analysis_json: str | None,
) -> dict:
    """Run Stages 7-8 for a completed photo: publish, archive, log.

    Returns the decision log entry (as a dict).
    """
    # Stage 7 — publish
    published_path = publish_done(photo_id, filename, image_path)

    # Stage 7 — archive RAW + RapidRaw sidecar (.rrdata) next to it
    archive_raw(raw_path)
    write_rapidraw_sidecar(filename, analysis_json)

    # Stage 8 — decision log
    log_entry = _decision_log(photo_id, filename, state, analysis_json)
    write_decision_log(log_entry)

    # Stage 8 — reclaim disk: the 16-bit scratch TIFFs are no longer needed
    cleanup_intermediates(Path(filename).stem)

    return log_entry
