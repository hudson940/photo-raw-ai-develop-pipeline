"""Publish a camera JPG as-is: no AI analysis, no develop, no retouch.

Camera JPGs are already a finished rendering, so by default they are only made viewable
and downloadable — preview, published output (the JPG itself, orientation applied) and
the archived source. A redo of the photo (web UI or `python -m pipeline.redo`) is what
develops/retouches it, and runs the AI analysis only when asked to. Set
PIPELINE_JPEG_AUTO_PROCESS=1 to send JPGs through the full pipeline like RAWs instead.

    python -m pipeline.jpeg_publish            # publish every JPG still waiting in the queue
"""

import logging
import shutil
import sys
from pathlib import Path

from PIL import Image

from . import db
from .config import CONFIG, is_jpeg
from .output import archive_raw, publish_done
from .preview import make_preview
from .storage import STORAGE

log = logging.getLogger("jpeg")


def publish_jpeg(conn, photo_id: int) -> Path:
    """Make a queued JPG photo viewable as-is and mark it done. Raises on failure."""
    row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"#{photo_id} not in the queue")
    src = Path(row["path"])
    if not src.exists():                       # already archived (re-publish)
        src = CONFIG.archive / row["filename"]
    if not src.exists():
        raise RuntimeError(f"source JPG for #{photo_id} not found")
    preview = make_preview(src)
    with Image.open(src) as im:
        orientation = im.getexif().get(0x0112, 1)
    if orientation in (0, 1):
        # upright already: publish the camera's own bytes (full quality, EXIF kept)
        out = CONFIG.output / f"{Path(row['filename']).stem}_{photo_id}.jpg"
        shutil.copyfile(src, out)
        STORAGE.put(out)
    else:
        # rotated by an EXIF flag: bake the rotation in, so every viewer and resize agrees
        out = publish_done(photo_id, row["filename"], src)
    archived = archive_raw(src) if src.parent != CONFIG.archive else src
    db.set_state(conn, photo_id, "done", preview_path=str(preview), error=None)
    log.info("#%d %s published as-is (no processing) -> %s", photo_id, row["filename"], out.name)
    return archived


def pending_jpegs(conn) -> list:
    """Queued JPGs that were never processed (no analysis yet)."""
    rows = conn.execute(
        "SELECT id, filename FROM photos WHERE state IN ('pending', 'previewed', 'failed')"
        " AND analysis_json IS NULL ORDER BY id").fetchall()
    return [r for r in rows if is_jpeg(Path(r["filename"]))]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-8s %(message)s",
                        datefmt="%H:%M:%S")
    conn = db.connect(CONFIG.db_path)
    rows = pending_jpegs(conn)
    ok = 0
    for r in rows:
        try:
            publish_jpeg(conn, r["id"])
            ok += 1
        except Exception as exc:
            log.error("#%d %s: %s", r["id"], r["filename"], exc)
    print(f"Published {ok}/{len(rows)} queued JPG(s) as-is")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
