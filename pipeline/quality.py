"""Pre-processing image quality gate.

Cheap, deterministic OpenCV checks on the preview JPEG — before the expensive
AI analysis + RAW develop — to catch shots not worth processing:

  - blur / soft focus       : variance of the Laplacian (low = blurry)
  - under-exposure          : dark mean + a large fraction of crushed shadows
  - over-exposure           : bright mean + a large fraction of blown highlights

`assess()` always returns the measured metrics (so the UI can show them); when
the gate is on and a photo fails, the worker skips processing it, parks it in the
'rejected' state, and defaults it to not-selected. Nothing here is generative —
same philosophy as the retouch engine.

Thresholds are tuned for the ~1536px preview and configurable via env
(PIPELINE_QUALITY_*). The Laplacian variance scales with resolution, so keep the
assessment on the preview (not the full render) for stable thresholds.
"""

import logging
from pathlib import Path

import numpy as np

from .config import CONFIG

log = logging.getLogger("quality")


def assess(image_path: Path) -> dict:
    """Return a quality report for a preview/JPEG image.

    {
      "passed": bool,                 # False when a hard flag fires and the gate is on
      "flags": ["blurry", "underexposed", ...],
      "blur": float,                  # variance of Laplacian
      "brightness": float,            # mean luma 0-255
      "shadow_clip": float,           # fraction of near-black pixels 0-1
      "highlight_clip": float,        # fraction of near-white pixels 0-1
      "reason": "short human summary",
    }
    """
    import cv2

    report = {"passed": True, "flags": [], "blur": 0.0, "brightness": 0.0,
              "shadow_clip": 0.0, "highlight_clip": 0.0, "reason": ""}
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            report["reason"] = "unreadable image"
            return report
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # focus: variance of the Laplacian, normalized to a 1000px long edge so the
        # threshold is resolution-independent across differently-sized previews
        h, w = gray.shape
        scale = 1000.0 / max(h, w)
        g = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else gray
        blur = float(cv2.Laplacian(g, cv2.CV_64F).var())

        brightness = float(gray.mean())
        total = gray.size
        shadow_clip = float((gray <= 8).sum()) / total
        highlight_clip = float((gray >= 247).sum()) / total

        report.update(blur=round(blur, 1), brightness=round(brightness, 1),
                      shadow_clip=round(shadow_clip, 4),
                      highlight_clip=round(highlight_clip, 4))

        flags = []
        if blur < CONFIG.quality_blur_min:
            flags.append("blurry")
        # under: dark overall AND lots of crushed shadow, or extremely dark
        if (brightness < CONFIG.quality_dark_max
                and shadow_clip > CONFIG.quality_clip_frac):
            flags.append("underexposed")
        # over: bright overall AND lots of blown highlight, or extremely bright
        if (brightness > CONFIG.quality_bright_min
                and highlight_clip > CONFIG.quality_clip_frac):
            flags.append("overexposed")
        report["flags"] = flags
        report["passed"] = not flags
        report["reason"] = _summary(report)
    except Exception as exc:                # a broken check must not block the pipeline
        log.warning("Quality assessment failed for %s: %s", image_path, exc)
        report["reason"] = f"assessment error: {exc}"
    return report


def _summary(r: dict) -> str:
    if not r["flags"]:
        return "ok"
    bits = []
    if "blurry" in r["flags"]:
        bits.append(f"blurry (focus {r['blur']:.0f})")
    if "underexposed" in r["flags"]:
        bits.append(f"underexposed ({r['shadow_clip']*100:.0f}% crushed)")
    if "overexposed" in r["flags"]:
        bits.append(f"overexposed ({r['highlight_clip']*100:.0f}% blown)")
    return ", ".join(bits)
