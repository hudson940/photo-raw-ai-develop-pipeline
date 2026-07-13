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
    # Big enough that a local reasoning model (Qwen3-VL etc.) can "think" and still answer.
    canary_max_tokens: int = int(os.environ.get("PIPELINE_CANARY_MAX_TOKENS", "1024"))
    # How hard the analysis model should reason: off | low | medium | high. Lower = faster.
    # On the Anthropic API this caps the thinking-token budget; on local OpenAI/Anthropic-
    # compatible servers (LM Studio/Qwen) it is sent as `reasoning_effort`. Defaults low so
    # local reasoning models don't burn time on long chains of thought for a structured task.
    reasoning_effort: str = os.environ.get("PIPELINE_REASONING_EFFORT", "low")
    # When True, the analysis preview has its background masked to neutral gray so the model
    # bases automatic exposure / white balance / color on the SUBJECT, not the background.
    analyze_subject_only: bool = os.environ.get("PIPELINE_ANALYZE_SUBJECT_ONLY", "").lower() in ("1", "true", "yes")
    # When True, the analysis preview shows only the SKIN (everything else masked gray) so the
    # model meters exposure on the skin — avoids over-exposing skin in low-key / bright-skin scenes.
    analyze_skin_exposure: bool = os.environ.get("PIPELINE_ANALYZE_SKIN_EXPOSURE", "").lower() in ("1", "true", "yes")

    # Watcher / worker behavior
    scan_interval_s: float = float(os.environ.get("PIPELINE_SCAN_INTERVAL", "5"))
    stable_check_delay_s: float = float(os.environ.get("PIPELINE_STABLE_DELAY", "2"))
    max_attempts: int = int(os.environ.get("PIPELINE_MAX_ATTEMPTS", "3"))
    retry_backoff_base_s: float = float(os.environ.get("PIPELINE_RETRY_BACKOFF", "30"))

    # Stage 4 — develop
    develop_output_bps: int = int(os.environ.get("PIPELINE_DEVELOP_BPS", "16"))
    develop_tiff_compression: str = os.environ.get("PIPELINE_TIFF_COMPRESSION", "zlib")
    # Highlights above this level roll off smoothly toward white instead of hard-clipping
    # (protects bright skin from blowing out). 1.0 disables the rolloff.
    highlight_rolloff_knee: float = float(os.environ.get("PIPELINE_HIGHLIGHT_KNEE", "0.75"))

    # Stage 7 — final delivery
    output_jpeg_quality: int = int(os.environ.get("PIPELINE_JPEG_QUALITY", "90"))
    keep_intermediate_tiffs: bool = os.environ.get("PIPELINE_KEEP_TIFFS", "").lower() in ("1", "true", "yes")

    # Stage 5-6 — retouch. birefnet-general is much more accurate for people/clothing than
    # u2net (which grays out dark gowns etc.); it's slower (~15s/photo on CPU) but only runs
    # when a mask op is used. Override with PIPELINE_REMBG_MODEL=u2net for speed.
    rembg_model: str = os.environ.get("PIPELINE_REMBG_MODEL", "birefnet-general")

    # Skin-tone warmth bias (-1..1): +ve pushes skin more orange, -ve leaves it yellower. Applied
    # by skin_tone_correction on top of the per-type natural hue target. Defaults to 0 (natural,
    # best-practice color for the skin type); raise it only if you want a warmer/orange look.
    skin_orange_warmth: float = float(os.environ.get("PIPELINE_SKIN_WARMTH", "0"))
    # Skin richness/saturation multiplier applied by skin_tone_correction (1.0 = leave chroma as
    # corrected, >1 = deeper/richer skin color so it doesn't look flat/washed out). Default 1.15.
    skin_saturation: float = float(os.environ.get("PIPELINE_SKIN_SATURATION", "1.15"))
    # Skin luminance lift (0..1) — brighten skin, with extra on lips/cheeks, like the Lightroom
    # Color-Mixer Orange/Red luminance sliders portrait editors use. 0 = no brightening.
    skin_luminance: float = float(os.environ.get("PIPELINE_SKIN_LUMINANCE", "0.45"))
    # Lip enhancement (0..1) — restore/deepen natural lip red + a little richness and definition
    # (skin correction can flatten lip colour). 0 = leave lips alone.
    lip_enhance: float = float(os.environ.get("PIPELINE_LIP_ENHANCE", "0.35"))
    # Skin red flush (0..1) — add healthy red to the skin midtones (cheeks), to match the warm red
    # of the camera's colour rendering that a flat RAW develop loses. 0 = no added red.
    skin_red: float = float(os.environ.get("PIPELINE_SKIN_RED", "0.35"))

    # Auto-levels (opt-in per-layer exposure): meter each region's midtone toward a target
    # and correct through the masks the retouch stage already builds. This is the reliable
    # alternative to asking a vision model to eyeball per-region exposure. Targets are 0-1
    # luminance; 0 disables that layer. Background defaults off (low/high-key is intentional).
    auto_levels_skin_target: float = float(os.environ.get("PIPELINE_AUTO_SKIN_TARGET", "0.72"))
    auto_levels_subject_target: float = float(os.environ.get("PIPELINE_AUTO_SUBJECT_TARGET", "0.5"))
    auto_levels_background_target: float = float(os.environ.get("PIPELINE_AUTO_BG_TARGET", "0"))
    # Per-layer correction is clamped to +/- this many stops so metering can never overcook.
    auto_levels_max_ev: float = float(os.environ.get("PIPELINE_AUTO_MAX_EV", "1.25"))

    # Auto background (--background auto): thresholds for deciding smooth/studio/keep. A backdrop
    # is "neutral" below this Lab chroma; wrinkles are detected when backdrop texture std (0-255)
    # exceeds the threshold. Raise the wrinkle threshold if clean backdrops get smoothed.
    auto_bg_neutral_chroma: float = float(os.environ.get("PIPELINE_AUTOBG_CHROMA", "12"))
    auto_bg_wrinkle_thresh: float = float(os.environ.get("PIPELINE_AUTOBG_WRINKLE", "4.0"))

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
