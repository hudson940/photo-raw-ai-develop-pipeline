"""Single-threaded worker: pending → previewed → analyzed → developed → retouched → done.

One photo at a time — GPU stages (darktable develop, ComfyUI retouch)
don't parallelize on a single card, so the queue discipline starts here.
Low-confidence analyses are routed to the 'review' state instead of continuing.
"""

import logging
import sqlite3
import time
from pathlib import Path

import anthropic

from .config import CONFIG
from . import db
from .analysis import analyze_preview, verify_vision
from .develop import develop, DevelopError
from .output import finalize
from .preview import make_preview
from .retouch import retouch

log = logging.getLogger("worker")


def _stage_develop(conn: sqlite3.Connection, photo_id: int, raw_path: Path, analysis_json: str) -> Path | None:
    """Run Stage 4: RAW development -> TIFF. Returns the TIFF path or None on failure."""
    try:
        tiff_path = develop(raw_path, analysis_json, CONFIG.work)
        db.set_state(conn, photo_id, "developed")
        log.info("#%d developed ok", photo_id)
        return tiff_path
    except DevelopError as exc:
        state = db.record_failure(
            conn, photo_id, f"DevelopError: {exc}",
            CONFIG.max_attempts, CONFIG.retry_backoff_base_s,
        )
        log.error("#%d develop: %s -> %s", photo_id, exc, state)
        return None
    except Exception as exc:
        state = db.record_failure(
            conn, photo_id, f"{type(exc).__name__}: {exc}",
            CONFIG.max_attempts, CONFIG.retry_backoff_base_s,
        )
        log.error("#%d develop unexpected: %s -> %s", photo_id, exc, state)
        return None


def _stage_retouch(conn: sqlite3.Connection, photo_id: int, tiff_path: Path, analysis_json: str) -> Path:
    """Run Stages 5-6: ComfyUI retouch. Returns the final image path."""
    try:
        final_path = retouch(tiff_path, analysis_json)
        db.set_state(conn, photo_id, "retouched")
        log.info("#%d retouched ok", photo_id)
        return final_path
    except Exception as exc:
        log.warning("#%d retouch failed (will use developed TIFF): %s", photo_id, exc)
        db.set_state(conn, photo_id, "retouched", error=f"retouch_skipped: {exc}")
        return tiff_path


def _stage_finalize(
    conn: sqlite3.Connection,
    photo_id: int,
    filename: str,
    raw_path: Path,
    image_path: Path,
    analysis_json: str | None,
) -> None:
    """Run Stages 7-8: publish, archive, decision log, mark done."""
    try:
        finalize(photo_id, filename, raw_path, image_path, "done", analysis_json)
        db.set_state(conn, photo_id, "done")
        log.info("#%d done — published to output/", photo_id)
    except Exception as exc:
        log.error("#%d finalize failed: %s", photo_id, exc)
        db.set_state(conn, photo_id, "done", error=f"finalize_partial: {exc}")


def process_one(conn: sqlite3.Connection, client: anthropic.Anthropic) -> bool:
    """Process the next pending photo through all stages. Returns False if queue empty."""
    row = db.claim_next_pending(conn)
    if row is None:
        return False

    photo_id, raw_path = row["id"], Path(row["path"])
    log.info("Processing #%d %s (attempt %d)", photo_id, raw_path.name, row["attempts"] + 1)

    if not raw_path.exists():
        db.set_state(conn, photo_id, "failed", error="source file disappeared from inbox")
        return True

    analysis_json: str | None = None
    preview_path: Path | None = None

    try:
        # If this is a retry after analysis succeeded (develop/retouch failed),
        # reuse the existing analysis_json and skip stages 2-3.
        analysis_json = row["analysis_json"]

        if analysis_json is None:
            # Stage 2 — preview (skip if a retry already produced one)
            preview_path = Path(row["preview_path"]) if row["preview_path"] else None
            if preview_path is None or not preview_path.exists():
                preview_path = make_preview(raw_path)
                db.set_state(conn, photo_id, "previewed", preview_path=str(preview_path))

            # Stage 3 — AI analysis
            result = analyze_preview(preview_path, client)
            analysis_json = result.model_dump_json()

            if result.confidence < CONFIG.confidence_threshold:
                db.set_state(
                    conn, photo_id, "review",
                    preview_path=str(preview_path),
                    analysis_json=analysis_json,
                    confidence=result.confidence,
                    error=f"confidence {result.confidence:.2f} < threshold {CONFIG.confidence_threshold}",
                )
                log.warning("#%d routed to review (confidence %.2f)", photo_id, result.confidence)
                return True

            db.set_state(
                conn, photo_id, "analyzed",
                preview_path=str(preview_path),
                analysis_json=analysis_json,
                confidence=result.confidence,
                error=None,
            )
            log.info("#%d analyzed ok — proceeding to develop", photo_id)
        else:
            log.info("#%d re-using cached analysis (retrying develop/retouch)", photo_id)
            preview_path = Path(row["preview_path"]) if row["preview_path"] else None

        # Stage 4 — develop (RAW -> TIFF)
        tiff_path = _stage_develop(conn, photo_id, raw_path, analysis_json)
        if tiff_path is None:
            return True  # develop failed, photo requeued as pending

        # Stages 5-6 — retouch (ComfyUI)
        final_image = _stage_retouch(conn, photo_id, tiff_path, analysis_json)

        # Stages 7-8 — output & archive
        _stage_finalize(conn, photo_id, raw_path.name, raw_path, final_image, analysis_json)

    except anthropic.AuthenticationError:
        db.set_state(conn, photo_id, "failed", error="ANTHROPIC_API_KEY invalid or missing")
        raise
    except anthropic.RateLimitError as exc:
        retry_after = float(exc.response.headers.get("retry-after", "60"))
        db.set_state(conn, photo_id, "pending", next_attempt_at=time.time() + retry_after)
        log.warning("Rate limited; retrying #%d in %.0fs", photo_id, retry_after)
        time.sleep(min(retry_after, 60))
    except Exception as exc:
        state = db.record_failure(
            conn, photo_id, f"{type(exc).__name__}: {exc}",
            CONFIG.max_attempts, CONFIG.retry_backoff_base_s,
        )
        log.error("#%d %s -> %s: %s", photo_id, raw_path.name, state, exc)

    return True


def run_forever(conn: sqlite3.Connection) -> None:
    client = anthropic.Anthropic()
    verify_vision(client)
    log.info("Worker started (model=%s, confidence threshold=%.2f)", CONFIG.model, CONFIG.confidence_threshold)
    while True:
        if not process_one(conn, client):
            time.sleep(CONFIG.scan_interval_s)
