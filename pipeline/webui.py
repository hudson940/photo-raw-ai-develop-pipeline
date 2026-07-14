"""Web UI + JSON API: browse developed photos, select a batch, re-render with a wizard.

    python -m pipeline.webui                      # http://127.0.0.1:8765
    python -m pipeline.webui --host 0.0.0.0       # reachable from the LAN

Serves a single-page gallery (pipeline/webui.html) backed by a small JSON API:

    GET  /api/photos            queue listing with render availability
    GET  /api/photos/{id}       full parameters + copy-paste reproduce command
    GET  /api/defaults          retouch defaults (used as wizard placeholders)
    GET  /thumb/{id}            cached thumbnail of the latest render
    GET  /img/{id}              full-size latest render (?src=preview for the analysis preview)
    POST /api/redo              {"ids": [...], "overrides": {...}} -> starts a render job
    GET  /api/jobs              job queue with per-photo progress
    POST /api/jobs/{id}/cancel  skip a job's remaining photos

Renders run sequentially in one background thread through the same code path as
`python -m pipeline.redo`, so output naming, sidecars and DB bookkeeping are identical.
No new dependencies: stdlib http.server + the pipeline itself.
"""

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from . import db
from .config import CONFIG
from .redo import _DEFAULT_RETOUCH, _process_photo, _reproduce_command

log = logging.getLogger("webui")

_HTML_PATH = Path(__file__).with_name("webui.html")
_THUMB_LONG_EDGE = 512
_THUMB_LOCK = threading.Lock()

# ---------------------------------------------------------------- overrides validation

# retouch float params the wizard may set -> allowed range
_R_FLOAT = {
    "skin_smoothing": (0, 1), "skin_tone_correction": (0, 1), "skin_warmth": (-1, 1),
    "skin_saturation": (0.3, 2.0), "skin_luminance": (0, 1), "skin_red": (0, 1),
    "reduce_dark_circles": (0, 1), "reduce_dewlap": (0, 1), "brighten_eyes": (0, 1),
    "iris_enhance": (0, 1), "whiten_teeth": (0, 1), "lip_enhance": (0, 1),
    "tame_highlights": (0, 1), "hair_texture": (0, 1), "hair_shimmer": (0, 1),
    "hair_defrizz": (0, 1), "clothing_contrast": (0, 1),
    "subject_exposure": (-3, 3), "background_exposure": (-3, 3),
    "auto_skin_target": (0, 1), "auto_subject_target": (0, 1), "auto_background_target": (0, 1),
}
_FACE_KEYS = {
    "skin_smoothing", "skin_tone_correction", "reduce_dark_circles", "reduce_dewlap",
    "brighten_eyes", "iris_enhance", "whiten_teeth", "lip_enhance", "tame_highlights",
    "hair_texture", "hair_shimmer", "hair_defrizz", "skin_luminance",
}
_SKIN_TYPES = {"auto", "fair", "light", "medium", "olive", "tan", "brown", "deep"}
_INTENSITIES = {"subtle", "natural", "polished"}
_BG_ACTIONS = {"auto", "keep", "blur", "smooth", "studio", "replace"}
_CLOTH_FLOAT = {"color_pop": (0, 1), "luminance": (-1, 1), "shadows": (-1, 1),
                "blacks": (-1, 1), "whites": (-1, 1)}

# ---------------------------------------------------------------- albums & share links

# Albums group photos for customer proofing. A share link is a random token + a
# password (HTTP Basic); its permission decides what the customer may do:
#   'select'  — view the album, mark photos selected/discarded
#   'develop' — the above plus the full redo wizard / crop / erase on album photos
_ALBUM_SCHEMA = """
CREATE TABLE IF NOT EXISTS albums (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS album_photos (
    album_id INTEGER NOT NULL,
    photo_id INTEGER NOT NULL,
    decision INTEGER,               -- NULL undecided / 1 selected / 0 discarded
    decided_at REAL,
    added_at REAL NOT NULL,
    PRIMARY KEY (album_id, photo_id)
);
CREATE TABLE IF NOT EXISTS album_shares (
    token TEXT PRIMARY KEY,
    album_id INTEGER NOT NULL,
    salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    permission TEXT NOT NULL DEFAULT 'select',   -- 'select' | 'develop'
    revoked INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""

_PBKDF2_ROUNDS = 200_000
_ADMIN_PASSWORD: str | None = None   # set from --admin-password / PIPELINE_WEBUI_PASSWORD


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return salt.hex(), dk.hex()


def _check_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    _, calc = _hash_password(password, salt_hex)
    return hmac.compare_digest(calc, hash_hex)


def _album_photo_decisions(conn, album_id: int) -> dict[int, int | None]:
    """photo_id -> decision (NULL/1/0) for every photo in an album."""
    return {r["photo_id"]: r["decision"] for r in conn.execute(
        "SELECT photo_id, decision FROM album_photos WHERE album_id=?", (album_id,))}


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def normalize_overrides(raw: dict) -> dict | None:
    """Whitelist + clamp a JSON overrides payload into the same shape (and with the
    same implication rules) that overrides_from_args builds for the CLI."""
    if not isinstance(raw, dict):
        return None
    rin = raw.get("retouch") or {}
    r: dict = {}

    for k, (lo, hi) in _R_FLOAT.items():
        if rin.get(k) is not None:
            r[k] = _clamp(rin[k], lo, hi)
    if rin.get("remove_blemishes") is not None:
        r["remove_blemishes"] = bool(rin["remove_blemishes"])
    if rin.get("skin_type") in _SKIN_TYPES:
        r["skin_type"] = rin["skin_type"]
    if rin.get("intensity") in _INTENSITIES:
        r["intensity"] = rin["intensity"]
    if rin.get("auto_levels") is not None:
        r["auto_levels"] = bool(rin["auto_levels"])

    cin = rin.get("clothing") or {}
    cl: dict = {}
    if cin.get("color"):
        cl["color"] = str(cin["color"]).strip().lower()[:24]
    for k, (lo, hi) in _CLOTH_FLOAT.items():
        if cin.get(k) is not None:
            cl[k] = _clamp(cin[k], lo, hi)
    if cl:
        r["clothing"] = cl

    ein = rin.get("erase") or {}
    er: dict = {}
    if "mask" in ein:   # empty string is meaningful: it clears a saved erase mask
        name = str(ein["mask"]).strip()
        # only bare filenames inside data/masks are accepted from the network
        er["mask"] = Path(name).name if name else ""
    if ein.get("method") in ("content-aware", "generative"):
        er["method"] = ein["method"]
    if ein.get("prompt"):
        er["prompt"] = str(ein["prompt"]).strip()[:500]
    if er:
        r["erase"] = er

    bin_ = rin.get("background") or {}
    bg: dict = {}
    if bin_.get("action") in _BG_ACTIONS:
        bg["action"] = bin_["action"]
    if bin_.get("color"):
        bg["color"] = str(bin_["color"]).strip()[:24]
        bg.setdefault("action", "studio")     # picking a color implies a studio backdrop
    if bin_.get("replace_prompt"):
        bg["replace_prompt"] = str(bin_["replace_prompt"]).strip()[:500]
    if bg:
        r["background"] = bg

    # implication rules, mirroring overrides_from_args
    if _FACE_KEYS & r.keys() or "remove_blemishes" in r:
        r["is_portrait"] = True
    for k in ("skin_type", "skin_warmth", "skin_saturation", "skin_red"):
        if k in r:
            r["is_portrait"] = True
            r.setdefault("skin_tone_correction", 0.35)  # these ride on tone correction
    if {"auto_skin_target", "auto_subject_target", "auto_background_target"} & r.keys():
        r["auto_levels"] = True

    win = raw.get("white_balance") or {}
    wb: dict = {}
    if win.get("mode") == "camera":
        wb["mode"] = "camera"
    elif win.get("mode") == "kelvin":
        wb["mode"] = "kelvin"
        if win.get("temp") is not None:
            wb["temp"] = int(_clamp(win["temp"], 2000, 50000))
        if win.get("tint") is not None:
            wb["tint"] = int(_clamp(win["tint"], -50, 50))

    din = raw.get("develop") or {}
    dv: dict = {}
    for k in ("shadows", "highlights"):
        if din.get(k) is not None:
            dv[k] = _clamp(din[k], -100, 100)
    cr = din.get("crop") or {}
    if all(k in cr for k in ("x", "y", "w", "h")):
        x = _clamp(cr["x"], 0, 1); y = _clamp(cr["y"], 0, 1)
        w = _clamp(cr["w"], 0.01, 1 - x); h = _clamp(cr["h"], 0.01, 1 - y)
        # a near-full-frame crop is a no-op reset; only send a real crop
        if not (x < 1e-3 and y < 1e-3 and w > 0.999 and h > 0.999):
            dv["crop"] = {"x": round(x, 5), "y": round(y, 5),
                          "w": round(w, 5), "h": round(h, 5), "aspect": "custom"}
        else:
            dv["crop"] = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "aspect": "original"}

    ov: dict = {}
    if r:
        ov["retouch"] = r
    if wb:
        ov["white_balance"] = wb
    if dv:
        ov["develop"] = dv
    return ov or None


# ---------------------------------------------------------------- render job queue

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[str]" = queue.Queue()


def submit_job(ids: list[int], overrides: dict | None, from_raw: bool, reanalyze: bool,
               subject_only: bool = False, skin_exposure: bool = False,
               share: str | None = None) -> dict:
    job = {
        "id": uuid.uuid4().hex[:8],
        "ids": ids,
        "overrides": overrides,
        "from_raw": from_raw,
        "reanalyze": reanalyze,
        "subject_only": subject_only,
        "skin_exposure": skin_exposure,
        "share": share,               # token of the share link that queued it, if any
        "state": "queued",
        "cancel": False,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "items": {pid: {"status": "queued", "error": None, "output": None} for pid in ids},
    }
    with _jobs_lock:
        _jobs[job["id"]] = job
    _job_queue.put(job["id"])
    log.info("Job %s queued: %d photo(s)%s", job["id"], len(ids),
             " (from RAW)" if from_raw else "")
    return job


def _job_public(job: dict) -> dict:
    """JSON-safe view of a job (drop the overrides payload duplication is fine to keep)."""
    return {
        "id": job["id"], "ids": job["ids"], "state": job["state"],
        "overrides": job["overrides"], "from_raw": job["from_raw"],
        "reanalyze": job["reanalyze"], "subject_only": job["subject_only"],
        "skin_exposure": job["skin_exposure"], "share": job.get("share"),
        "created_at": job["created_at"],
        "started_at": job["started_at"], "finished_at": job["finished_at"],
        "items": {str(k): v for k, v in job["items"].items()},
    }


def _render_worker() -> None:
    """Single sequential worker: renders are CPU/RAM heavy, one at a time is the point."""
    conn = db.connect(CONFIG.db_path)
    while True:
        job_id = _job_queue.get()
        job = _jobs[job_id]
        job["state"] = "running"
        job["started_at"] = time.time()
        args = SimpleNamespace(reanalyze=job["reanalyze"], from_raw=job["from_raw"])
        # analysis-time metering choices (only matter with reanalyze), like the
        # --subject-only / --skin-exposure CLI flags
        CONFIG.analyze_subject_only = bool(job["subject_only"])
        CONFIG.analyze_skin_exposure = bool(job["skin_exposure"])
        for pid in job["ids"]:
            item = job["items"][pid]
            if job["cancel"]:
                item["status"] = "cancelled"
                continue
            item["status"] = "running"
            try:
                out_path = _process_photo(conn, pid, args, job["overrides"])
                item["status"] = "done"
                item["output"] = out_path.name
                log.info("Job %s #%d -> %s", job_id, pid, out_path.name)
            except Exception as exc:  # keep the batch going, record the failure
                item["status"] = "error"
                item["error"] = str(exc)
                log.error("Job %s #%d failed: %s", job_id, pid, exc)
        job["state"] = "cancelled" if job["cancel"] else "done"
        job["finished_at"] = time.time()


# ---------------------------------------------------------------- photo helpers

def _output_path(photo_id: int, filename: str) -> Path:
    return CONFIG.output / f"{Path(filename).stem}_{photo_id}.jpg"


def _photo_row(conn, photo_id: int):
    return conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()


def _image_source(row, prefer_preview: bool = False) -> Path | None:
    """Best available JPEG for a photo: the published render, else the analysis preview."""
    out = _output_path(row["id"], row["filename"])
    preview = Path(row["preview_path"]) if row["preview_path"] else None
    candidates = [preview, out] if prefer_preview else [out, preview]
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def _thumb_path(row) -> Path | None:
    """Return a cached thumbnail path, (re)building it if the source render is newer."""
    src = _image_source(row)
    if src is None:
        return None
    tdir = CONFIG.root / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    thumb = tdir / f"{row['id']}.jpg"
    if thumb.exists() and thumb.stat().st_mtime >= src.stat().st_mtime:
        return thumb
    with _THUMB_LOCK:
        if thumb.exists() and thumb.stat().st_mtime >= src.stat().st_mtime:
            return thumb
        img = ImageOps.exif_transpose(Image.open(src))
        img.thumbnail((_THUMB_LONG_EDGE, _THUMB_LONG_EDGE))
        tmp = thumb.with_suffix(".tmp.jpg")
        img.convert("RGB").save(tmp, "JPEG", quality=82)
        tmp.replace(thumb)
    return thumb


def _list_photos(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, filename, state, confidence, updated_at, preview_path, analysis_json"
        " FROM photos ORDER BY id"
    ).fetchall()
    photos = []
    for r in rows:
        analysis = json.loads(r["analysis_json"]) if r["analysis_json"] else {}
        rp = analysis.get("retouch") or {}
        out = _output_path(r["id"], r["filename"])
        has_output = out.exists()
        photos.append({
            "id": r["id"],
            "filename": r["filename"],
            "state": r["state"],
            "confidence": r["confidence"],
            "is_portrait": bool(rp.get("is_portrait")),
            "scene": (analysis.get("scene_description") or "")[:160],
            "has_analysis": bool(r["analysis_json"]),
            "has_output": has_output,
            "output_mtime": int(out.stat().st_mtime) if has_output else 0,
            "has_preview": bool(r["preview_path"] and Path(r["preview_path"]).exists()),
        })
    return photos


def _photo_detail(conn, photo_id: int) -> dict | None:
    row = _photo_row(conn, photo_id)
    if row is None:
        return None
    analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else {}
    analysis["retouch"] = {**_DEFAULT_RETOUCH, **(analysis.get("retouch") or {})}
    return {
        "id": row["id"],
        "filename": row["filename"],
        "state": row["state"],
        "confidence": row["confidence"],
        "scene": analysis.get("scene_description") or "",
        "develop": analysis.get("develop") or {},
        "retouch": analysis["retouch"],
        "has_analysis": bool(row["analysis_json"]),
        "command": _reproduce_command(row["id"], analysis) if row["analysis_json"] else None,
    }


# ---------------------------------------------------------------- HTTP handler

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # --- plumbing -------------------------------------------------------
    def log_message(self, fmt, *args):  # route access logs to logging (debug level)
        log.debug("%s %s", self.address_string(), fmt % args)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str, cache: bool = False) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=31536000, immutable" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    # --- auth -------------------------------------------------------------
    def _basic_password(self) -> str | None:
        """Password from an HTTP Basic Authorization header (username is ignored)."""
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return None
        try:
            return base64.b64decode(auth[6:].strip()).decode("utf-8", "replace").partition(":")[2]
        except Exception:
            return None

    def _unauthorized(self, realm: str) -> None:
        body = b"Password required."
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{realm}", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _admin_ok(self) -> bool:
        """Gate for the operator UI/API. Open unless --admin-password is set."""
        if not _ADMIN_PASSWORD:
            return True
        pw = self._basic_password()
        if pw is not None and hmac.compare_digest(pw, _ADMIN_PASSWORD):
            return True
        self._unauthorized("PhotoRAW operator")
        return False

    def _share_auth(self, token: str):
        """Return the share row when the Basic-auth password matches, else answer
        401 (or 404 for an unknown/revoked link) and return None."""
        share = self._conn().execute(
            "SELECT * FROM album_shares WHERE token=? AND revoked=0", (token,)).fetchone()
        if share is None:
            self._error(404, "this link is no longer valid")
            return None
        pw = self._basic_password()
        if pw is not None and _check_password(pw, share["salt"], share["password_hash"]):
            return share
        self._unauthorized(f"album-{token[:8]}")   # per-link realm so creds don't collide
        return None

    # --- routes ---------------------------------------------------------
    def _serve_thumb(self, photo_id: int, url) -> None:
        row = _photo_row(self._conn(), photo_id)
        thumb = _thumb_path(row) if row else None
        if thumb is None:
            return self._error(404, "no image for this photo yet")
        # URL carries ?v=<mtime>, so the content is immutable per URL
        return self._file(thumb, "image/jpeg", cache="v" in parse_qs(url.query))

    def _serve_img(self, photo_id: int, url) -> None:
        row = _photo_row(self._conn(), photo_id)
        prefer_preview = parse_qs(url.query).get("src", [""])[0] == "preview"
        src = _image_source(row, prefer_preview) if row else None
        if src is None:
            return self._error(404, "no image for this photo yet")
        return self._file(src, "image/jpeg")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if m := re.fullmatch(r"/share/([A-Za-z0-9_-]+)(/.*)?", url.path):
                return self._share_get(m.group(1), m.group(2) or "/", url)
            if not self._admin_ok():
                return
            if url.path in ("/", "/index.html"):
                return self._file(_HTML_PATH, "text/html; charset=utf-8")
            if url.path == "/api/photos":
                return self._json({"photos": _list_photos(self._conn())})
            if m := re.fullmatch(r"/api/photos/(\d+)", url.path):
                detail = _photo_detail(self._conn(), int(m.group(1)))
                return self._json(detail) if detail else self._error(404, "photo not found")
            if url.path == "/api/defaults":
                return self._json({"retouch": _DEFAULT_RETOUCH})
            if url.path == "/api/albums":
                return self._albums_list()
            if m := re.fullmatch(r"/api/albums/(\d+)", url.path):
                return self._album_detail(int(m.group(1)))
            if url.path == "/api/jobs":
                with _jobs_lock:
                    jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
                    return self._json({"jobs": [_job_public(j) for j in jobs[:30]]})
            if m := re.fullmatch(r"/thumb/(\d+)", url.path):
                return self._serve_thumb(int(m.group(1)), url)
            if m := re.fullmatch(r"/img/(\d+)", url.path):
                return self._serve_img(int(m.group(1)), url)
            return self._error(404, "not found")
        except BrokenPipeError:
            pass
        except Exception as exc:
            log.exception("GET %s failed", self.path)
            self._error(500, str(exc))

    def _share_get(self, token: str, sub: str, url) -> None:
        share = self._share_auth(token)
        if share is None:
            return
        conn = self._conn()
        decisions = _album_photo_decisions(conn, share["album_id"])
        if sub in ("", "/"):
            album = conn.execute("SELECT name FROM albums WHERE id=?",
                                 (share["album_id"],)).fetchone()
            cfg = {"prefix": f"/share/{token}", "permission": share["permission"],
                   "album": album["name"] if album else "Album"}
            html = _HTML_PATH.read_text()
            html = html.replace(
                "<script>", f"<script>window.SHARE = {json.dumps(cfg)};</script>\n<script>", 1)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if sub == "/api/photos":
            photos = [p for p in _list_photos(conn) if p["id"] in decisions]
            for p in photos:
                p["decision"] = decisions[p["id"]]
                p.pop("scene", None)
                p.pop("confidence", None)
            return self._json({"photos": photos})
        if m := re.fullmatch(r"/thumb/(\d+)", sub):
            pid = int(m.group(1))
            if pid not in decisions:
                return self._error(404, "not in this album")
            return self._serve_thumb(pid, url)
        if m := re.fullmatch(r"/img/(\d+)", sub):
            pid = int(m.group(1))
            if pid not in decisions:
                return self._error(404, "not in this album")
            return self._serve_img(pid, url)
        # everything below is the develop surface
        if share["permission"] != "develop":
            return self._error(403, "this link only allows selecting or discarding photos")
        if sub == "/api/defaults":
            return self._json({"retouch": _DEFAULT_RETOUCH})
        if m := re.fullmatch(r"/api/photos/(\d+)", sub):
            pid = int(m.group(1))
            if pid not in decisions:
                return self._error(404, "not in this album")
            detail = _photo_detail(conn, pid)
            return self._json(detail) if detail else self._error(404, "photo not found")
        if sub == "/api/jobs":
            with _jobs_lock:
                jobs = [j for j in _jobs.values() if j.get("share") == token]
            jobs.sort(key=lambda j: j["created_at"], reverse=True)
            return self._json({"jobs": [_job_public(j) for j in jobs[:30]]})
        return self._error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if m := re.fullmatch(r"/share/([A-Za-z0-9_-]+)(/.+)", url.path):
                return self._share_post(m.group(1), m.group(2))
            if not self._admin_ok():
                return
            if url.path == "/api/redo":
                return self._redo()
            if m := re.fullmatch(r"/api/photos/(\d+)/erase", url.path):
                return self._erase(int(m.group(1)))
            if url.path == "/api/albums":
                return self._album_create()
            if m := re.fullmatch(r"/api/albums/(\d+)/photos", url.path):
                return self._album_edit_photos(int(m.group(1)))
            if m := re.fullmatch(r"/api/albums/(\d+)/delete", url.path):
                return self._album_delete(int(m.group(1)))
            if m := re.fullmatch(r"/api/albums/(\d+)/shares", url.path):
                return self._share_create(int(m.group(1)))
            if m := re.fullmatch(r"/api/shares/([A-Za-z0-9_-]+)/revoke", url.path):
                return self._share_revoke(m.group(1))
            if m := re.fullmatch(r"/api/jobs/([0-9a-f]+)/cancel", url.path):
                return self._job_cancel(m.group(1), share_token=None)
            return self._error(404, "not found")
        except BrokenPipeError:
            pass
        except Exception as exc:
            log.exception("POST %s failed", self.path)
            self._error(500, str(exc))

    def _job_cancel(self, job_id: str, share_token: str | None) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
        # share links may only touch their own jobs
        if job is None or (share_token is not None and job.get("share") != share_token):
            return self._error(404, "job not found")
        job["cancel"] = True
        if job["state"] == "queued":
            job["state"] = "cancelled"
        return self._json({"ok": True, "job": _job_public(job)})

    def _share_post(self, token: str, sub: str) -> None:
        share = self._share_auth(token)
        if share is None:
            return
        conn = self._conn()
        decisions = _album_photo_decisions(conn, share["album_id"])
        if sub == "/api/decision":
            body = self._read_json()
            if body is None:
                return self._error(400, "invalid JSON body")
            try:
                pid = int(body.get("photo_id"))
            except (TypeError, ValueError):
                return self._error(400, "photo_id must be an integer")
            if pid not in decisions:
                return self._error(404, "not in this album")
            val = {"select": 1, "discard": 0, "clear": None}.get(body.get("decision"), "bad")
            if val == "bad":
                return self._error(400, "decision must be select, discard or clear")
            with conn:
                conn.execute(
                    "UPDATE album_photos SET decision=?, decided_at=?"
                    " WHERE album_id=? AND photo_id=?",
                    (val, time.time(), share["album_id"], pid))
            return self._json({"ok": True, "photo_id": pid, "decision": val})
        if share["permission"] != "develop":
            return self._error(403, "this link only allows selecting or discarding photos")
        if sub == "/api/redo":
            return self._redo(share_token=token, allowed_ids=set(decisions))
        if m := re.fullmatch(r"/api/photos/(\d+)/erase", sub):
            pid = int(m.group(1))
            if pid not in decisions:
                return self._error(404, "not in this album")
            return self._erase(pid, share_token=token)
        if m := re.fullmatch(r"/api/jobs/([0-9a-f]+)/cancel", sub):
            return self._job_cancel(m.group(1), share_token=token)
        return self._error(404, "not found")

    def _redo(self, share_token: str | None = None,
              allowed_ids: set[int] | None = None) -> None:
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        try:
            ids = [int(i) for i in body.get("ids", [])]
        except (TypeError, ValueError):
            return self._error(400, "ids must be a list of integers")
        if not ids:
            return self._error(400, "no photo ids given")
        if allowed_ids is not None:
            outside = [i for i in ids if i not in allowed_ids]
            if outside:
                return self._error(403, f"photos not in this album: {outside}")
        conn = self._conn()
        missing = [i for i in ids if _photo_row(conn, i) is None]
        if missing:
            return self._error(400, f"unknown photo ids: {missing}")
        overrides = normalize_overrides(body.get("overrides") or {})
        job = submit_job(ids, overrides,
                         from_raw=bool(body.get("from_raw")),
                         # re-analysis spends the operator's API credits — operator only
                         reanalyze=bool(body.get("reanalyze")) and share_token is None,
                         subject_only=bool(body.get("subject_only")),
                         skin_exposure=bool(body.get("skin_exposure")),
                         share=share_token)
        self._json({"job": _job_public(job)}, 202)

    def _erase(self, photo_id: int, share_token: str | None = None) -> None:
        """Save an operator-drawn erase mask (data-URL PNG, white = remove) for a photo
        and start a single-photo render with it. {"clear": true} removes a saved mask."""
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        row = _photo_row(self._conn(), photo_id)
        if row is None:
            return self._error(404, "photo not found")

        masks_dir = CONFIG.root / "masks"
        mask_name = f"{photo_id}.png"
        if body.get("clear"):
            (masks_dir / mask_name).unlink(missing_ok=True)
            erase = {"mask": ""}
        else:
            data_url = body.get("mask") or ""
            m = re.fullmatch(r"data:image/png;base64,([A-Za-z0-9+/=\s]+)", data_url)
            if not m:
                return self._error(400, "mask must be a data:image/png;base64 URL")
            masks_dir.mkdir(parents=True, exist_ok=True)
            (masks_dir / mask_name).write_bytes(base64.b64decode(m.group(1)))
            erase = {"mask": mask_name,
                     "method": body.get("method", "content-aware"),
                     "prompt": body.get("prompt", "")}

        overrides = normalize_overrides({"retouch": {"erase": erase}})
        if not body.get("render", True):
            return self._json({"ok": True, "erase": erase})
        job = submit_job([photo_id], overrides, from_raw=False, reanalyze=False,
                         share=share_token)
        self._json({"job": _job_public(job)}, 202)

    # --- albums (operator) -------------------------------------------------
    def _albums_list(self) -> None:
        conn = self._conn()
        albums = []
        for a in conn.execute("SELECT * FROM albums ORDER BY id DESC"):
            c = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(decision=1),0) AS sel,"
                " COALESCE(SUM(decision=0),0) AS dis FROM album_photos WHERE album_id=?",
                (a["id"],)).fetchone()
            shares = [{"token": s["token"], "permission": s["permission"],
                       "url": f"/share/{s['token']}", "created_at": s["created_at"]}
                      for s in conn.execute(
                          "SELECT * FROM album_shares WHERE album_id=? AND revoked=0"
                          " ORDER BY created_at", (a["id"],))]
            albums.append({"id": a["id"], "name": a["name"], "created_at": a["created_at"],
                           "count": c["n"], "selected": c["sel"], "discarded": c["dis"],
                           "shares": shares})
        self._json({"albums": albums})

    def _album_detail(self, album_id: int) -> None:
        conn = self._conn()
        a = conn.execute("SELECT * FROM albums WHERE id=?", (album_id,)).fetchone()
        if a is None:
            return self._error(404, "album not found")
        photos = [{"photo_id": r["photo_id"], "decision": r["decision"],
                   "decided_at": r["decided_at"]}
                  for r in conn.execute(
                      "SELECT * FROM album_photos WHERE album_id=? ORDER BY photo_id",
                      (album_id,))]
        self._json({"id": a["id"], "name": a["name"], "photos": photos})

    def _album_create(self) -> None:
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        name = str(body.get("name") or "").strip()[:80]
        if not name:
            return self._error(400, "album name required")
        try:
            ids = [int(i) for i in body.get("photo_ids") or []]
        except (TypeError, ValueError):
            return self._error(400, "photo_ids must be a list of integers")
        conn = self._conn()
        now = time.time()
        with conn:
            cur = conn.execute("INSERT INTO albums (name, created_at) VALUES (?, ?)",
                               (name, now))
            album_id = cur.lastrowid
            conn.executemany(
                "INSERT OR IGNORE INTO album_photos (album_id, photo_id, added_at)"
                " VALUES (?,?,?)",
                [(album_id, pid, now) for pid in ids])
        log.info("Album #%d %r created with %d photo(s)", album_id, name, len(ids))
        self._json({"id": album_id, "name": name, "count": len(ids)}, 201)

    def _album_edit_photos(self, album_id: int) -> None:
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        conn = self._conn()
        if conn.execute("SELECT 1 FROM albums WHERE id=?", (album_id,)).fetchone() is None:
            return self._error(404, "album not found")
        add = [int(i) for i in body.get("add") or []]
        remove = [int(i) for i in body.get("remove") or []]
        now = time.time()
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO album_photos (album_id, photo_id, added_at)"
                " VALUES (?,?,?)",
                [(album_id, pid, now) for pid in add])
            conn.executemany(
                "DELETE FROM album_photos WHERE album_id=? AND photo_id=?",
                [(album_id, pid) for pid in remove])
        n = conn.execute("SELECT COUNT(*) AS n FROM album_photos WHERE album_id=?",
                         (album_id,)).fetchone()["n"]
        self._json({"ok": True, "count": n})

    def _album_delete(self, album_id: int) -> None:
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM album_photos WHERE album_id=?", (album_id,))
            conn.execute("DELETE FROM album_shares WHERE album_id=?", (album_id,))
            cur = conn.execute("DELETE FROM albums WHERE id=?", (album_id,))
        if cur.rowcount == 0:
            return self._error(404, "album not found")
        self._json({"ok": True})

    def _share_create(self, album_id: int) -> None:
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        conn = self._conn()
        if conn.execute("SELECT 1 FROM albums WHERE id=?", (album_id,)).fetchone() is None:
            return self._error(404, "album not found")
        password = str(body.get("password") or "")
        if len(password) < 4:
            return self._error(400, "password must be at least 4 characters")
        permission = body.get("permission") or "select"
        if permission not in ("select", "develop"):
            return self._error(400, "permission must be 'select' or 'develop'")
        token = secrets.token_urlsafe(12)
        salt, pw_hash = _hash_password(password)
        with conn:
            conn.execute(
                "INSERT INTO album_shares (token, album_id, salt, password_hash,"
                " permission, created_at) VALUES (?,?,?,?,?,?)",
                (token, album_id, salt, pw_hash, permission, time.time()))
        log.info("Share link for album #%d created (%s)", album_id, permission)
        self._json({"token": token, "url": f"/share/{token}", "permission": permission}, 201)

    def _share_revoke(self, token: str) -> None:
        conn = self._conn()
        with conn:
            cur = conn.execute("UPDATE album_shares SET revoked=1 WHERE token=?", (token,))
        if cur.rowcount == 0:
            return self._error(404, "share link not found")
        self._json({"ok": True})

    def _conn(self):
        # one short-lived connection per request; SQLite in WAL mode handles this fine
        if not hasattr(self, "_db"):
            self._db = db.connect(CONFIG.db_path)
        return self._db


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-8s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--admin-password", default=os.environ.get("PIPELINE_WEBUI_PASSWORD", ""),
                   help="require this password (HTTP Basic) for the operator UI/API. Share "
                        "links always use their own per-link passwords. Strongly recommended "
                        "when binding beyond localhost (--host 0.0.0.0)")
    args = p.parse_args()

    global _ADMIN_PASSWORD
    _ADMIN_PASSWORD = args.admin_password or None
    if args.host != "127.0.0.1" and not _ADMIN_PASSWORD:
        log.warning("Binding to %s WITHOUT --admin-password: anyone on the network can "
                    "use the operator UI. Share links still require their passwords.", args.host)

    conn = db.connect(CONFIG.db_path)
    conn.executescript(_ALBUM_SCHEMA)
    conn.close()

    threading.Thread(target=_render_worker, daemon=True, name="render-worker").start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Photo review UI on http://%s:%d  (output: %s)", args.host, args.port, CONFIG.output)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("bye")


if __name__ == "__main__":
    main()
