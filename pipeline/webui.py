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

from . import auth, db
from .config import CONFIG, RAW_EXTENSIONS
from .redo import _DEFAULT_RETOUCH, _find_raw, _process_photo, _reproduce_command
from .storage import STORAGE

log = logging.getLogger("webui")

# Built React SPA (Vite -> web/dist). Falls back to the legacy single-file UI when the
# SPA hasn't been built (e.g. a source checkout without `npm run build`).
_DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"
_SPA_INDEX = _DIST_DIR / "index.html"
_LEGACY_HTML = Path(__file__).with_name("webui.html")
_HTML_PATH = _SPA_INDEX if _SPA_INDEX.exists() else _LEGACY_HTML

_ASSET_TYPES = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
                ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon",
                ".woff2": "font/woff2", ".woff": "font/woff", ".json": "application/json",
                ".map": "application/json", ".webp": "image/webp"}
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
    owner TEXT,                     -- editor who owns this album (NULL = admin/shared)
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


def _init_album_schema(conn) -> None:
    conn.executescript(_ALBUM_SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(albums)")}
    if "owner" not in have:
        with conn:
            conn.execute("ALTER TABLE albums ADD COLUMN owner TEXT")


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
    if din.get("rotation_deg") is not None:
        # straighten (fine) + 90° orientation, combined into one angle; normalize to -180..180
        deg = float(din["rotation_deg"]) % 360.0
        if deg > 180.0:
            deg -= 360.0
        dv["rotation_deg"] = round(deg, 3)
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
               share: str | None = None, user: str | None = None) -> dict:
    job = {
        "id": uuid.uuid4().hex[:8],
        "ids": ids,
        "overrides": overrides,
        "from_raw": from_raw,
        "reanalyze": reanalyze,
        "subject_only": subject_only,
        "skin_exposure": skin_exposure,
        "share": share,               # token of the share link that queued it, if any
        "user": user,                 # operator username that queued it (multi-tenancy)
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
        "user": job.get("user"), "created_at": job["created_at"],
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


def _photo_row_by_name(conn, filename: str):
    return conn.execute("SELECT * FROM photos WHERE filename = ?", (filename,)).fetchone()


def _image_source(row, prefer_preview: bool = False) -> Path | None:
    """Best available JPEG for a photo: the published render, else the analysis preview.
    Falls back to object storage (downloading into the local cache) when configured."""
    out = _output_path(row["id"], row["filename"])
    preview = Path(row["preview_path"]) if row["preview_path"] else None
    candidates = [preview, out] if prefer_preview else [out, preview]
    for p in candidates:
        if p is not None and STORAGE.get(p) is not None:
            return p
    return None


def _has_geometry(analysis: dict) -> bool:
    """True if the stored develop already applies a crop or rotation."""
    dp = (analysis or {}).get("develop") or {}
    cr = dp.get("crop") or {}
    cropped = all(k in cr for k in ("x", "y", "w", "h")) and not (
        cr["x"] < 1e-3 and cr["y"] < 1e-3 and cr["w"] > 0.999 and cr["h"] > 0.999)
    return cropped or abs(float(dp.get("rotation_deg", 0) or 0)) > 0.01


def _base_image(row) -> Path | None:
    """Full-frame image in the DEVELOPED orientation — the correct base for the crop/
    straighten editor (the analysis preview can disagree with rawpy on orientation).

    When the photo has no crop/rotation yet, the published render already IS the full
    developed frame, so serve it. Otherwise re-develop the RAW with crop/rotation
    stripped (cached) so the editor and the backend share one coordinate space."""
    if not row["analysis_json"]:
        return _image_source(row)
    analysis = json.loads(row["analysis_json"])
    if not _has_geometry(analysis):
        return _image_source(row)          # output == full developed frame already
    cache = CONFIG.work / f"base_{row['id']}.jpg"
    if cache.exists():
        return cache
    raw = _find_raw(row)
    if raw is None:
        return _image_source(row)
    import copy
    import tempfile
    from .develop import develop
    from .output import _to_jpeg
    stripped = copy.deepcopy(analysis)
    dp = stripped.setdefault("develop", {})
    dp["crop"] = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "aspect": "original"}
    dp["rotation_deg"] = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tiff = develop(raw, json.dumps(stripped), Path(tmp))
        CONFIG.work.mkdir(parents=True, exist_ok=True)
        _to_jpeg(tiff, cache, CONFIG.output_jpeg_quality)
    log.info("Built full-frame crop base for #%d", row["id"])
    return cache


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
    STORAGE.put(thumb)
    return thumb


def _list_photos(conn, owner: str | None = None) -> list[dict]:
    """List photos, optionally scoped to a single owner (editor multi-tenancy).
    owner=None means no filter (super admin / auth disabled)."""
    sql = ("SELECT id, filename, state, confidence, updated_at, preview_path,"
           " analysis_json, owner, quality_json, selected FROM photos")
    params: tuple = ()
    if owner is not None:
        sql += " WHERE owner = ?"
        params = (owner,)
    sql += " ORDER BY id"
    rows = conn.execute(sql, params).fetchall()
    photos = []
    for r in rows:
        analysis = json.loads(r["analysis_json"]) if r["analysis_json"] else {}
        rp = analysis.get("retouch") or {}
        quality = json.loads(r["quality_json"]) if r["quality_json"] else None
        out = _output_path(r["id"], r["filename"])
        has_output = STORAGE.exists(out)
        photos.append({
            "id": r["id"],
            "filename": r["filename"],
            "state": r["state"],
            "confidence": r["confidence"],
            "owner": r["owner"],
            "selected": r["selected"],
            "quality_flags": (quality or {}).get("flags") or [],
            "quality_reason": (quality or {}).get("reason") or "",
            "is_portrait": bool(rp.get("is_portrait")),
            "scene": (analysis.get("scene_description") or "")[:160],
            "has_analysis": bool(r["analysis_json"]),
            "has_output": has_output,
            "output_mtime": int(STORAGE.mtime(out)) if has_output else 0,
            "has_preview": bool(r["preview_path"] and STORAGE.exists(Path(r["preview_path"]))),
        })
    return photos


def _photo_visible(row, user: dict | None) -> bool:
    """Editors may only touch their own photos; super admin (or auth-off) sees all."""
    if row is None:
        return False
    if user is None or auth.is_super_admin(user):
        return True
    return row["owner"] == user["username"]


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

    @property
    def _extra_headers(self) -> list:
        if not hasattr(self, "_extra"):
            self._extra: list = []
        return self._extra

    def _emit_extra(self) -> None:
        for k, v in self._extra_headers:
            self.send_header(k, v)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._emit_extra()
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str, cache: bool = False) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=31536000, immutable" if cache else "no-store")
        self._emit_extra()
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
        """Password-fallback gate (no Keycloak): open unless --admin-password is set."""
        if not _ADMIN_PASSWORD:
            return True
        pw = self._basic_password()
        if pw is not None and hmac.compare_digest(pw, _ADMIN_PASSWORD):
            return True
        self._unauthorized("PhotoRAW operator")
        return False

    def _cookies(self) -> dict:
        jar = {}
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k:
                jar[k] = v
        return jar

    def _set_session_cookie(self, token: str) -> None:
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
        self._extra_headers.append(
            ("Set-Cookie",
             f"pr_session={token}; HttpOnly; SameSite=Lax; Path=/; "
             f"Max-Age={CONFIG.session_ttl_s}{secure}"))

    def _clear_session_cookie(self) -> None:
        self._extra_headers.append(
            ("Set-Cookie", "pr_session=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"))

    def _current_user(self) -> dict | None:
        """The logged-in operator, or None. Keycloak: from the session cookie.
        No Keycloak but a password set: Basic-auth acts as an implicit super admin.
        Neither: open dev mode -> implicit super admin."""
        if auth.enabled():
            return auth.read_session(self._cookies().get("pr_session"))
        if _ADMIN_PASSWORD:
            pw = self._basic_password()
            if pw is None or not hmac.compare_digest(pw, _ADMIN_PASSWORD):
                return None
        return {"username": "admin", "name": "Administrator", "roles": [auth.SUPER_ADMIN]}

    def _require_operator(self) -> dict | None:
        """Return the current operator or answer 401 and return None."""
        user = self._current_user()
        if user is not None:
            return user
        if auth.enabled() or not _ADMIN_PASSWORD:
            self._error(401, "authentication required")   # SPA shows the login form
        else:
            self._unauthorized("PhotoRAW operator")        # Basic-auth challenge
        return None

    @staticmethod
    def _owner_scope(user: dict) -> str | None:
        """Owner filter for queries: None for super admin (see all), else the username."""
        return None if auth.is_super_admin(user) else user["username"]

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
    def _serve_index(self, share_cfg: dict | None = None) -> None:
        """Serve the SPA's index.html with a runtime config injected as window.__CONFIG__.
        Works for the built React app and the legacy single-file UI alike."""
        html = _HTML_PATH.read_text()
        cfg = {"share": share_cfg} if share_cfg else {}
        inject = f"<script>window.__CONFIG__={json.dumps(cfg)};</script>"
        # legacy file reads window.SHARE; keep it working too
        if share_cfg:
            inject += f"<script>window.SHARE={json.dumps(share_cfg)};</script>"
        html = html.replace("<head>", "<head>" + inject, 1) if "<head>" in html \
            else inject + html
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_asset(self, rel: str) -> None:
        """Serve a built static asset from web/dist (Vite output). 404 if missing."""
        target = (_DIST_DIR / rel).resolve()
        if not str(target).startswith(str(_DIST_DIR.resolve())) or not target.is_file():
            return self._error(404, "not found")
        ctype = _ASSET_TYPES.get(target.suffix.lower(), "application/octet-stream")
        # hashed asset filenames are immutable; index/other served no-store implicitly
        self._file(target, ctype, cache=target.parent.name == "assets")

    def _serve_thumb(self, photo_id: int, url) -> None:
        row = _photo_row(self._conn(), photo_id)
        thumb = _thumb_path(row) if row else None
        if thumb is None:
            return self._error(404, "no image for this photo yet")
        # URL carries ?v=<mtime>, so the content is immutable per URL
        return self._file(thumb, "image/jpeg", cache="v" in parse_qs(url.query))

    def _serve_img(self, photo_id: int, url) -> None:
        row = _photo_row(self._conn(), photo_id)
        which = parse_qs(url.query).get("src", [""])[0]
        if row is None:
            return self._error(404, "no image for this photo yet")
        if which == "base":                       # full-frame developed image for the crop editor
            src = _base_image(row)
        else:
            src = _image_source(row, prefer_preview=which == "preview")
        if src is None:
            return self._error(404, "no image for this photo yet")
        return self._file(src, "image/jpeg")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if m := re.fullmatch(r"/share/([A-Za-z0-9_-]+)(/.*)?", url.path):
                return self._share_get(m.group(1), m.group(2) or "/", url)
            # --- public (no session needed) ---
            if url.path in ("/", "/index.html"):
                return self._serve_index()
            if m := re.fullmatch(r"/(assets/.+|favicon\.[a-z]+|[\w.-]+\.(?:svg|png|ico|webp|woff2?|js|css))",
                                 url.path):
                return self._serve_asset(m.group(1))
            if url.path == "/api/me":
                user = self._current_user()
                return self._json({"authenticated": user is not None, "auth": auth.enabled(),
                                   "user": user})
            # --- operator (session/password required) ---
            user = self._require_operator()
            if user is None:
                return
            owner = self._owner_scope(user)
            if url.path == "/api/photos":
                return self._json({"photos": _list_photos(self._conn(), owner)})
            if m := re.fullmatch(r"/api/photos/(\d+)", url.path):
                row = _photo_row(self._conn(), int(m.group(1)))
                if not _photo_visible(row, user):
                    return self._error(404, "photo not found")
                return self._json(_photo_detail(self._conn(), int(m.group(1))))
            if url.path == "/api/defaults":
                return self._json({"retouch": _DEFAULT_RETOUCH})
            if url.path == "/api/albums":
                return self._albums_list(owner)
            if m := re.fullmatch(r"/api/albums/(\d+)", url.path):
                return self._album_detail(int(m.group(1)), user)
            if url.path == "/api/users":
                return self._users_list(user)
            if url.path == "/api/jobs":
                uname = None if auth.is_super_admin(user) else user["username"]
                with _jobs_lock:
                    jobs = [j for j in _jobs.values()
                            if j.get("share") is None
                            and (uname is None or j.get("user") == uname)]
                jobs.sort(key=lambda j: j["created_at"], reverse=True)
                return self._json({"jobs": [_job_public(j) for j in jobs[:30]]})
            if m := re.fullmatch(r"/thumb/(\d+)", url.path):
                if not _photo_visible(_photo_row(self._conn(), int(m.group(1))), user):
                    return self._error(404, "no image for this photo yet")
                return self._serve_thumb(int(m.group(1)), url)
            if m := re.fullmatch(r"/img/(\d+)", url.path):
                if not _photo_visible(_photo_row(self._conn(), int(m.group(1))), user):
                    return self._error(404, "no image for this photo yet")
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
            return self._serve_index(share_cfg={
                "prefix": f"/share/{token}", "permission": share["permission"],
                "album": album["name"] if album else "Album"})
        if m := re.fullmatch(r"/(assets/.+|[\w.-]+\.(?:svg|png|ico|webp|woff2?|js|css))", sub):
            return self._serve_asset(m.group(1))   # share pages load the same bundle
        if sub == "/api/photos":
            photos = [p for p in _list_photos(conn) if p["id"] in decisions]
            for p in photos:
                p["decision"] = decisions[p["id"]]
                for k in ("scene", "confidence", "owner", "selected"):
                    p.pop(k, None)
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
            # --- public auth endpoints ---
            if url.path == "/auth/login":
                return self._login()
            if url.path == "/auth/logout":
                self._clear_session_cookie()
                return self._json({"ok": True})
            # --- operator (session/password required) ---
            user = self._require_operator()
            if user is None:
                return
            if url.path == "/api/redo":
                return self._redo(user=user)
            if url.path == "/api/upload":
                return self._upload(user)
            if m := re.fullmatch(r"/api/photos/(\d+)/erase", url.path):
                return self._erase(int(m.group(1)), user=user)
            if m := re.fullmatch(r"/api/photos/(\d+)/select", url.path):
                return self._set_selected(int(m.group(1)), user)
            if m := re.fullmatch(r"/api/photos/(\d+)/process", url.path):
                return self._force_process(int(m.group(1)), user)
            if url.path == "/api/albums":
                return self._album_create(user)
            if m := re.fullmatch(r"/api/albums/(\d+)/photos", url.path):
                return self._album_edit_photos(int(m.group(1)), user)
            if m := re.fullmatch(r"/api/albums/(\d+)/delete", url.path):
                return self._album_delete(int(m.group(1)), user)
            if m := re.fullmatch(r"/api/albums/(\d+)/shares", url.path):
                return self._share_create(int(m.group(1)), user)
            if m := re.fullmatch(r"/api/shares/([A-Za-z0-9_-]+)/revoke", url.path):
                return self._share_revoke(m.group(1), user)
            if m := re.fullmatch(r"/api/jobs/([0-9a-f]+)/cancel", url.path):
                return self._job_cancel(m.group(1), share_token=None)
            # --- user management (super admin only) ---
            if url.path == "/api/users":
                return self._user_create(user)
            if m := re.fullmatch(r"/api/users/([0-9a-f-]+)/role", url.path):
                return self._user_role(m.group(1), user)
            if m := re.fullmatch(r"/api/users/([0-9a-f-]+)/enabled", url.path):
                return self._user_enabled(m.group(1), user)
            if m := re.fullmatch(r"/api/users/([0-9a-f-]+)/delete", url.path):
                return self._user_delete(m.group(1), user)
            return self._error(404, "not found")
        except BrokenPipeError:
            pass
        except Exception as exc:
            log.exception("POST %s failed", self.path)
            self._error(500, str(exc))

    # --- auth endpoints -------------------------------------------------
    def _login(self) -> None:
        if not auth.enabled():
            return self._error(400, "Keycloak login is not configured on this server")
        body = self._read_json() or {}
        username, password = str(body.get("username", "")), str(body.get("password", ""))
        if not username or not password:
            return self._error(400, "username and password required")
        try:
            user = auth.login(username, password)
        except auth.AuthError as exc:
            return self._error(401, str(exc))
        self._set_session_cookie(auth.make_session(user))
        self._json({"ok": True, "user": user})

    def _require_super_admin(self, user: dict) -> bool:
        if auth.is_super_admin(user):
            return True
        self._error(403, "super admin only")
        return False

    def _users_list(self, user: dict) -> None:
        if not self._require_super_admin(user):
            return
        if not auth.enabled():
            return self._json({"users": [], "auth": False})
        try:
            return self._json({"users": auth.list_users(), "auth": True})
        except auth.AuthError as exc:
            return self._error(502, str(exc))

    def _user_create(self, user: dict) -> None:
        if not self._require_super_admin(user):
            return
        body = self._read_json() or {}
        try:
            created = auth.create_user(
                str(body.get("username", "")).strip(), str(body.get("password", "")),
                body.get("role", auth.EDITOR), str(body.get("email", "")).strip(),
                temporary=bool(body.get("temporary")))
        except auth.AuthError as exc:
            return self._error(400, str(exc))
        self._json({"ok": True, "user": created}, 201)

    def _user_role(self, uid: str, user: dict) -> None:
        if not self._require_super_admin(user):
            return
        try:
            auth.set_user_role(uid, (self._read_json() or {}).get("role", ""))
        except auth.AuthError as exc:
            return self._error(400, str(exc))
        self._json({"ok": True})

    def _user_enabled(self, uid: str, user: dict) -> None:
        if not self._require_super_admin(user):
            return
        try:
            auth.set_user_enabled(uid, bool((self._read_json() or {}).get("enabled", True)))
        except auth.AuthError as exc:
            return self._error(400, str(exc))
        self._json({"ok": True})

    def _user_delete(self, uid: str, user: dict) -> None:
        if not self._require_super_admin(user):
            return
        try:
            auth.delete_user(uid)
        except auth.AuthError as exc:
            return self._error(400, str(exc))
        self._json({"ok": True})

    def _upload(self, user: dict) -> None:
        """Accept a RAW file upload and drop it in the uploader's inbox subfolder so the
        watcher ingests it with the right owner. The raw file bytes are the request body;
        the filename comes from ?filename=… (avoids multipart parsing in the stdlib server).
        Editors upload into inbox/<username>/; super admin / open mode into the inbox root."""
        from . import watcher
        params = parse_qs(urlparse(self.path).query)
        raw_name = (params.get("filename", [""])[0] or "").strip()
        name = Path(raw_name).name                      # strip any path components
        if not name:
            return self._error(400, "filename query parameter required")
        if Path(name).suffix.lower() not in RAW_EXTENSIONS:
            return self._error(415, "only RAW files are accepted (e.g. .CR3, .NEF, .ARW, .DNG)")
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0:
            return self._error(400, "empty upload")
        if length > CONFIG.max_upload_mb * 1024 * 1024:
            return self._error(413, f"file exceeds the {CONFIG.max_upload_mb} MB limit")

        owner = None if auth.is_super_admin(user) else user["username"]
        dest_dir = CONFIG.inbox / owner if owner else CONFIG.inbox
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.exists() or _photo_row_by_name(self._conn(), name) is not None:
            return self._error(409, f"a photo named {name} already exists")

        # stream the body to a temp file, then atomically move into the inbox
        tmp = dest.with_name(dest.name + ".part")
        remaining, chunk = length, 1024 * 256
        try:
            with open(tmp, "wb") as f:
                while remaining > 0:
                    buf = self.rfile.read(min(chunk, remaining))
                    if not buf:
                        break
                    f.write(buf)
                    remaining -= len(buf)
            tmp.replace(dest)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            return self._error(500, f"upload failed: {exc}")
        if db.enqueue(self._conn(), dest, owner):
            log.info("Uploaded %s (owner=%s) -> queued", name, owner or "-")
        self._json({"ok": True, "filename": name, "owner": owner}, 201)

    # --- operator photo actions -----------------------------------------
    def _set_selected(self, photo_id: int, user: dict) -> None:
        row = _photo_row(self._conn(), photo_id)
        if not _photo_visible(row, user):
            return self._error(404, "photo not found")
        val = {"select": 1, "deselect": 0, "clear": None}.get(
            (self._read_json() or {}).get("selected"), "bad")
        if val == "bad":
            return self._error(400, "selected must be select, deselect or clear")
        db.set_state(self._conn(), photo_id, row["state"], selected=val)
        self._json({"ok": True, "photo_id": photo_id, "selected": val})

    def _force_process(self, photo_id: int, user: dict) -> None:
        """Force a quality-rejected (or any) photo through develop despite the gate,
        by queueing a render with re-analysis (which bypasses the worker's quality gate)."""
        row = _photo_row(self._conn(), photo_id)
        if not _photo_visible(row, user):
            return self._error(404, "photo not found")
        job = submit_job([photo_id], None, from_raw=True, reanalyze=True,
                         user=user["username"])
        self._json({"job": _job_public(job)}, 202)

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
              allowed_ids: set[int] | None = None, user: dict | None = None) -> None:
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
        # editors may only redo their own photos
        if user is not None and not auth.is_super_admin(user):
            outside = [i for i in ids if not _photo_visible(_photo_row(conn, i), user)]
            if outside:
                return self._error(403, f"not your photos: {outside}")
        overrides = normalize_overrides(body.get("overrides") or {})
        job = submit_job(ids, overrides,
                         from_raw=bool(body.get("from_raw")),
                         # re-analysis spends the operator's API credits — operator only
                         reanalyze=bool(body.get("reanalyze")) and share_token is None,
                         subject_only=bool(body.get("subject_only")),
                         skin_exposure=bool(body.get("skin_exposure")),
                         share=share_token, user=(user or {}).get("username"))
        self._json({"job": _job_public(job)}, 202)

    def _erase(self, photo_id: int, share_token: str | None = None,
               user: dict | None = None) -> None:
        """Save an operator-drawn erase mask (data-URL PNG, white = remove) for a photo
        and start a single-photo render with it. {"clear": true} removes a saved mask."""
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        row = _photo_row(self._conn(), photo_id)
        if row is None:
            return self._error(404, "photo not found")
        if user is not None and not _photo_visible(row, user):
            return self._error(404, "photo not found")

        masks_dir = CONFIG.root / "masks"
        mask_name = f"{photo_id}.png"
        if body.get("clear"):
            STORAGE.delete(masks_dir / mask_name)
            erase = {"mask": ""}
        else:
            data_url = body.get("mask") or ""
            m = re.fullmatch(r"data:image/png;base64,([A-Za-z0-9+/=\s]+)", data_url)
            if not m:
                return self._error(400, "mask must be a data:image/png;base64 URL")
            masks_dir.mkdir(parents=True, exist_ok=True)
            (masks_dir / mask_name).write_bytes(base64.b64decode(m.group(1)))
            STORAGE.put(masks_dir / mask_name)
            erase = {"mask": mask_name,
                     "method": body.get("method", "content-aware"),
                     "prompt": body.get("prompt", "")}

        overrides = normalize_overrides({"retouch": {"erase": erase}})
        if not body.get("render", True):
            return self._json({"ok": True, "erase": erase})
        job = submit_job([photo_id], overrides, from_raw=False, reanalyze=False,
                         share=share_token, user=(user or {}).get("username"))
        self._json({"job": _job_public(job)}, 202)

    # --- albums (operator) -------------------------------------------------
    def _albums_list(self, owner: str | None = None) -> None:
        conn = self._conn()
        albums = []
        sql = "SELECT * FROM albums"
        params: tuple = ()
        if owner is not None:
            sql += " WHERE owner = ?"
            params = (owner,)
        sql += " ORDER BY id DESC"
        for a in conn.execute(sql, params):
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

    def _album_owned(self, conn, album_id: int, user: dict):
        """Fetch an album row the user may access, else None (after writing 404)."""
        a = conn.execute("SELECT * FROM albums WHERE id=?", (album_id,)).fetchone()
        if a is None or (not auth.is_super_admin(user) and a["owner"] != user["username"]):
            self._error(404, "album not found")
            return None
        return a

    def _album_detail(self, album_id: int, user: dict) -> None:
        conn = self._conn()
        a = self._album_owned(conn, album_id, user)
        if a is None:
            return
        photos = [{"photo_id": r["photo_id"], "decision": r["decision"],
                   "decided_at": r["decided_at"]}
                  for r in conn.execute(
                      "SELECT * FROM album_photos WHERE album_id=? ORDER BY photo_id",
                      (album_id,))]
        self._json({"id": a["id"], "name": a["name"], "photos": photos})

    def _owned_ids(self, conn, ids: list[int], user: dict) -> list[int]:
        """Keep only photo ids the user may use (all, for super admin)."""
        if auth.is_super_admin(user):
            return ids
        return [i for i in ids if _photo_visible(_photo_row(conn, i), user)]

    def _album_create(self, user: dict) -> None:
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
        ids = self._owned_ids(conn, ids, user)          # can't album someone else's photos
        now = time.time()
        with conn:
            cur = conn.execute("INSERT INTO albums (name, owner, created_at) VALUES (?, ?, ?)",
                               (name, self._owner_scope(user), now))
            album_id = cur.lastrowid
            conn.executemany(
                "INSERT OR IGNORE INTO album_photos (album_id, photo_id, decision, added_at)"
                " VALUES (?,?,?,?)",
                [(album_id, pid, self._default_decision(conn, pid), now) for pid in ids])
        log.info("Album #%d %r created with %d photo(s)", album_id, name, len(ids))
        self._json({"id": album_id, "name": name, "count": len(ids)}, 201)

    @staticmethod
    def _default_decision(conn, photo_id: int):
        """Photos the operator/quality gate marked not-selected start discarded in albums."""
        row = conn.execute("SELECT selected FROM photos WHERE id=?", (photo_id,)).fetchone()
        return 0 if (row and row["selected"] == 0) else None

    def _album_edit_photos(self, album_id: int, user: dict) -> None:
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        conn = self._conn()
        if self._album_owned(conn, album_id, user) is None:
            return
        add = self._owned_ids(conn, [int(i) for i in body.get("add") or []], user)
        remove = [int(i) for i in body.get("remove") or []]
        now = time.time()
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO album_photos (album_id, photo_id, decision, added_at)"
                " VALUES (?,?,?,?)",
                [(album_id, pid, self._default_decision(conn, pid), now) for pid in add])
            conn.executemany(
                "DELETE FROM album_photos WHERE album_id=? AND photo_id=?",
                [(album_id, pid) for pid in remove])
        n = conn.execute("SELECT COUNT(*) AS n FROM album_photos WHERE album_id=?",
                         (album_id,)).fetchone()["n"]
        self._json({"ok": True, "count": n})

    def _album_delete(self, album_id: int, user: dict) -> None:
        conn = self._conn()
        if self._album_owned(conn, album_id, user) is None:
            return
        with conn:
            conn.execute("DELETE FROM album_photos WHERE album_id=?", (album_id,))
            conn.execute("DELETE FROM album_shares WHERE album_id=?", (album_id,))
            conn.execute("DELETE FROM albums WHERE id=?", (album_id,))
        self._json({"ok": True})

    def _share_create(self, album_id: int, user: dict) -> None:
        body = self._read_json()
        if body is None:
            return self._error(400, "invalid JSON body")
        conn = self._conn()
        if self._album_owned(conn, album_id, user) is None:
            return
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

    def _share_revoke(self, token: str, user: dict) -> None:
        conn = self._conn()
        share = conn.execute("SELECT album_id FROM album_shares WHERE token=?", (token,)).fetchone()
        if share is None or self._album_owned(conn, share["album_id"], user) is None:
            if share is not None:      # _album_owned already answered 404
                return
            return self._error(404, "share link not found")
        with conn:
            conn.execute("UPDATE album_shares SET revoked=1 WHERE token=?", (token,))
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
    if auth.enabled():
        log.info("Auth: Keycloak at %s (realm %s) — operators log in with their accounts",
                 CONFIG.keycloak_url, CONFIG.keycloak_realm)
    elif _ADMIN_PASSWORD:
        log.info("Auth: single admin password (set KEYCLOAK_URL for multi-user + roles)")
    elif args.host != "127.0.0.1":
        log.warning("Binding to %s with NO authentication: anyone on the network can use the "
                    "operator UI. Set KEYCLOAK_URL or PIPELINE_WEBUI_PASSWORD.", args.host)

    conn = db.connect(CONFIG.db_path)
    _init_album_schema(conn)
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
