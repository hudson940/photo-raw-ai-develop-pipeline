"""Stages 7-8: Output orchestration, archive, decision log, and review gate.

Consumes rows in the developed/retouched state and:
- Copies the final image to output/ with an organised name
- Moves the RAW to archive/
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
    """Publish the final image to output/ as a JPEG. Returns the output path."""
    stem = Path(filename).stem
    out_name = f"{stem}_{_timestamp()}.jpg"
    out_path = CONFIG.output / out_name
    _to_jpeg(image_path, out_path, CONFIG.output_jpeg_quality)
    size_mb = out_path.stat().st_size / 1e6
    log.info("Published %s -> %s (%.1f MB, q%d)", image_path.name, out_name, size_mb,
             CONFIG.output_jpeg_quality)
    return out_path


def archive_raw(raw_path: Path) -> Path:
    """Move a processed RAW to the archive directory."""
    dest = CONFIG.archive / raw_path.name
    if raw_path.exists():
        shutil.move(str(raw_path), str(dest))
        log.info("Archived %s", raw_path.name)
    return dest


def write_decision_log(log_entry: dict) -> Path:
    """Append a decision entry to the pipeline decision log."""
    log_path = CONFIG.root / "decision_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
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

    # Stage 7 — archive RAW
    archive_raw(raw_path)

    # Stage 8 — decision log
    log_entry = _decision_log(photo_id, filename, state, analysis_json)
    write_decision_log(log_entry)

    # Stage 8 — reclaim disk: the 16-bit scratch TIFFs are no longer needed
    cleanup_intermediates(Path(filename).stem)

    return log_entry
