"""Central configuration. Every value can be overridden via environment variable.

A `.env` file in the project root (KEY=VALUE lines) is loaded first — the
standard place for ANTHROPIC_API_KEY so the pipeline works from any shell,
cron job, or systemd unit. Real environment variables take precedence.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = _PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".3fr",
    ".erf", ".kdc", ".mos", ".mrw", ".x3f", ".iiq",
}


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


@dataclass
class Config:
    root: Path = field(default_factory=lambda: _env_path("PIPELINE_ROOT", _PROJECT_ROOT / "data"))

    # AI analysis
    model: str = os.environ.get("PIPELINE_MODEL", "deepseek-v4-flash")
    confidence_threshold: float = float(os.environ.get("PIPELINE_CONFIDENCE_THRESHOLD", "0.6"))
    preview_long_edge: int = int(os.environ.get("PIPELINE_PREVIEW_LONG_EDGE", "1536"))

    # Watcher / worker behavior
    scan_interval_s: float = float(os.environ.get("PIPELINE_SCAN_INTERVAL", "5"))
    stable_check_delay_s: float = float(os.environ.get("PIPELINE_STABLE_DELAY", "2"))
    max_attempts: int = int(os.environ.get("PIPELINE_MAX_ATTEMPTS", "3"))
    retry_backoff_base_s: float = float(os.environ.get("PIPELINE_RETRY_BACKOFF", "30"))

    # Stage 4 — develop
    develop_output_bps: int = int(os.environ.get("PIPELINE_DEVELOP_BPS", "16"))
    develop_tiff_compression: str = os.environ.get("PIPELINE_TIFF_COMPRESSION", "zlib")

    # Stage 7 — final delivery
    output_jpeg_quality: int = int(os.environ.get("PIPELINE_JPEG_QUALITY", "90"))
    keep_intermediate_tiffs: bool = os.environ.get("PIPELINE_KEEP_TIFFS", "").lower() in ("1", "true", "yes")

    # Stage 5-6 — retouch
    rembg_model: str = os.environ.get("PIPELINE_REMBG_MODEL", "u2net")

    # ComfyUI (used only for generative background replacement)
    comfyui_url: str = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
    comfyui_timeout: int = int(os.environ.get("COMFYUI_TIMEOUT", "600"))
    comfyui_checkpoint: str = os.environ.get("COMFYUI_CHECKPOINT", "v1-5-pruned-emaonly.safetensors")
    skip_retouch: bool = os.environ.get("PIPELINE_SKIP_RETOUCH", "").lower() in ("1", "true", "yes")

    @property
    def inbox(self) -> Path:
        return _env_path("PIPELINE_INBOX", self.root / "inbox")

    @property
    def previews(self) -> Path:
        return self.root / "previews"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def work(self) -> Path:
        """Scratch space for intermediate 16-bit TIFFs (develop + retouch)."""
        return self.root / "work"

    @property
    def archive(self) -> Path:
        return self.root / "archive"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def db_path(self) -> Path:
        return self.root / "pipeline.db"

    def ensure_dirs(self) -> None:
        for d in (self.inbox, self.previews, self.output, self.work, self.archive, self.review):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
