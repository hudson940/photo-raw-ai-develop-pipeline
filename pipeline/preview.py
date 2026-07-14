"""Stage 2: RAW -> downscaled preview JPEG for the AI.

Fast path extracts the embedded JPEG with exiftool; first fallback decodes the
RAW with rawpy (pure pip, no root needed); last resort is darktable-cli.
Output is capped at CONFIG.preview_long_edge.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from .config import CONFIG
from .storage import STORAGE

log = logging.getLogger("preview")

# largest-first: full-size embedded JPEG, then the smaller preview variants
EXIFTOOL_PREVIEW_TAGS = ["JpgFromRaw", "PreviewImage", "OtherImage", "ThumbnailImage"]


class PreviewError(Exception):
    pass


def find_tool(name: str) -> str | None:
    """Resolve a tool on PATH, falling back to ~/.local/bin (user-local installs)."""
    found = shutil.which(name)
    if found:
        return found
    local = Path.home() / ".local" / "bin" / name
    return str(local) if local.exists() else None


def _extract_with_exiftool(raw_path: Path, dest: Path) -> bool:
    exiftool = find_tool("exiftool")
    if exiftool is None:
        return False
    for tag in EXIFTOOL_PREVIEW_TAGS:
        result = subprocess.run(
            [exiftool, "-b", f"-{tag}", str(raw_path)],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and len(result.stdout) > 10_000:  # tiny thumbnails are useless
            dest.write_bytes(result.stdout)
            return True
    return False


def _render_with_rawpy(raw_path: Path, dest: Path) -> bool:
    try:
        import rawpy
    except ImportError:
        return False
    try:
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
        Image.fromarray(rgb).save(dest, "JPEG", quality=92)
        return True
    except Exception as exc:
        log.warning("rawpy could not decode %s: %s", raw_path.name, exc)
        return False


def _render_with_darktable(raw_path: Path, dest: Path) -> bool:
    darktable = find_tool("darktable-cli")
    if darktable is None:
        return False
    result = subprocess.run(
        [
            darktable, str(raw_path), str(dest),
            "--width", str(CONFIG.preview_long_edge),
            "--height", str(CONFIG.preview_long_edge),
            "--core", "--library", ":memory:",
        ],
        capture_output=True, timeout=300,
    )
    if result.returncode != 0:
        log.warning("darktable-cli failed for %s: %s", raw_path.name, result.stderr.decode(errors="replace")[-500:])
    return result.returncode == 0 and dest.exists()


def _downscale(src: Path, dest: Path) -> None:
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        img.thumbnail((CONFIG.preview_long_edge, CONFIG.preview_long_edge), Image.LANCZOS)
        img.convert("RGB").save(dest, "JPEG", quality=85)


def make_preview(raw_path: Path) -> Path:
    """Produce previews/<stem>.jpg for the given RAW file."""
    dest = CONFIG.previews / f"{raw_path.stem}.jpg"
    with tempfile.TemporaryDirectory(dir=CONFIG.previews) as tmp_dir:
        # a path that does NOT exist yet — darktable-cli refuses to overwrite
        # an existing file and silently exports to <name>_01.jpg instead
        tmp_path = Path(tmp_dir) / "full.jpg"
        for produce in (_extract_with_exiftool, _render_with_rawpy, _render_with_darktable):
            if produce is _render_with_rawpy:
                log.info("No embedded preview in %s, decoding the RAW directly", raw_path.name)
            if produce(raw_path, tmp_path):
                _downscale(tmp_path, dest)
                STORAGE.put(dest)
                return dest
    raise PreviewError(f"Could not produce a preview for {raw_path}")
