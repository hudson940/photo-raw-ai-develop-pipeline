"""Stage 4: RAW development -> 16-bit TIFF.

Decodes the RAW with rawpy (or a camera JPG with PIL), applies the AI-determined develop parameters
(exposure, white balance, contrast, highlights/shadows, saturation/vibrance,
crop, rotation), and exports a 16-bit TIFF ready for Stage 5 retouch or
final delivery.
"""

import io
import json
import logging
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageCms, ImageOps

from .config import CONFIG, is_jpeg
from .preview import find_tool

log = logging.getLogger("develop")


class DevelopError(Exception):
    pass


def _kelvin_to_rgb_multipliers(kelvin: float) -> tuple[float, float, float]:
    """Approximate RGB multipliers for a given color temperature (Kelvin).

    Based on Tanner Helland's algorithm. Returns multipliers relative to
    6500 K (neutral daylight), so a neutral white balance at 5500 K will
    produce multipliers slightly away from (1,1,1).
    """
    t = kelvin / 100.0

    if t <= 66:
        r = 1.0
        g = max(0, 99.4708025861 * math.log(t) - 161.1195681661) / 255.0
        g = max(0.0, min(1.0, g))
        if t <= 19:
            b = 0.0
        else:
            b = max(0, 138.5177312231 * math.log(t - 10) - 305.0447927307) / 255.0
            b = max(0.0, min(1.0, b))
    else:
        r = max(0, 329.698727446 * ((t - 60) ** -0.1332047592)) / 255.0
        r = max(0.0, min(1.0, r))
        g = max(0, 288.1221695283 * ((t - 60) ** -0.0755148492)) / 255.0
        g = max(0.0, min(1.0, g))
        b = 1.0

    # Normalise so D65 (6500 K) is the identity (1, 1, 1)
    d65_t = 6500 / 100.0
    d65_r = max(0, 329.698727446 * ((d65_t - 60) ** -0.1332047592)) / 255.0
    d65_g = max(0, 288.1221695283 * ((d65_t - 60) ** -0.0755148492)) / 255.0
    d65_b = 1.0

    return (r / d65_r, g / d65_g, b / d65_b)


def _wb_relative_multipliers(kelvin: float) -> tuple[float, float, float]:
    """WB adjustment relative to the camera's as-shot balance (pivot 6500 K = no change).

    Uses the inverse of the blackbody color so the control behaves like a normal
    photo-editor temperature slider: higher Kelvin -> warmer (more red, less blue),
    lower -> cooler. Green is held at 1.0 so overall brightness is roughly preserved.
    """
    def inv(t: float) -> tuple[float, float, float]:
        r, g, b = _kelvin_to_rgb_multipliers(t)
        return (1.0 / max(r, 1e-3), 1.0 / max(g, 1e-3), 1.0 / max(b, 1e-3))

    ir, ig, ib = inv(kelvin)
    pr, pg, pb = inv(6500)                          # pin the pivot so 6500 K = identity
    r, g, b = ir / pr, ig / pg, ib / pb
    return (r / g, 1.0, b / g)


def _highlight_rolloff(img: np.ndarray, knee: float) -> np.ndarray:
    """Soft-clip highlights: values above `knee` roll off toward 1.0 (asymptotically) instead
    of hard-clipping, so bright skin/specular keeps gradation rather than blowing to white.
    knee=1.0 disables it (plain clip)."""
    img = np.clip(img, 0.0, None)
    if knee >= 1.0:
        return np.clip(img, 0, 1)
    hi = img > knee
    span = 1.0 - knee
    img[hi] = knee + span * (1.0 - np.exp(-(img[hi] - knee) / span))
    return np.clip(img, 0, 1)


def _apply_tint(r: float, g: float, b: float, tint: float) -> tuple[float, float, float]:
    """Shift green-magenta balance. tint in [-50, 50], 0 = neutral."""
    factor = 10 ** (tint / 200.0)
    return (r, g * factor, b)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep clamped to [0, 1]: 0 at 0, 1 at 1, with zero slope at both
    ends — so region weights fade in/out with no kink (the source of the old banding)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _tone_curve(img: np.ndarray, contrast: float, highlights: float, shadows: float) -> np.ndarray:
    """Apply a contrast S-curve with highlight/shadow controls.

    Operates on 0-1 float image. contrast in [-1, 1], highlights/shadows in [-100, 100].

    Shadows/highlights use smooth, monotonic region weights so the transform never
    inverts or plateaus. The black and white points are preserved (a shadow lift
    never turns black gray, a highlight recovery never turns white gray), and the
    weight fades to zero at the midtone pivot with matching slope, so smooth skin /
    background gradients don't band. Positive = lift shadows / brighten highlights;
    negative = crush shadows / recover highlights.
    """
    # Convert to float in 0-1
    was_int = img.dtype != np.float32
    if was_int:
        max_val = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / max_val

    # Contrast S-curve: blend between identity and sigmoid
    if abs(contrast) > 0.001:
        pivot = 0.5
        t = (img - pivot) * (1.0 + contrast * 2.0) + pivot
        img = np.clip(t, 0, 1)

    # Shadows: scale the lower half by a smooth weight (1 at black -> 0 at the pivot).
    # Multiplicative, so pure black stays black and the map is monotonic.
    if abs(shadows) > 0.01:
        s = shadows / 100.0
        w = 1.0 - _smoothstep(img / 0.5)
        img = img * (1.0 + s * w)

    # Highlights: work on the headroom to white (1 - img) with a smooth weight
    # (0 at the pivot -> 1 at white). Positive shrinks the headroom (brighter),
    # negative grows it (recover); pure white stays white and the map is monotonic.
    if abs(highlights) > 0.01:
        h = highlights / 100.0
        w = _smoothstep((img - 0.5) / 0.5)
        img = 1.0 - (1.0 - img) * (1.0 - h * w)

    img = np.clip(img, 0, 1)

    if was_int:
        img = (img * max_val).astype(np.uint16)
    return img


def _apply_saturation_vibrance(
    img: np.ndarray, saturation: float, vibrance: float
) -> np.ndarray:
    """Adjust saturation and vibrance on a 0-1 float RGB image."""
    was_int = img.dtype != np.float32
    if was_int:
        max_val = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / max_val

    if abs(saturation) < 0.001 and abs(vibrance) < 0.001:
        return img

    # RGB -> HSL (simple luminance-based conversion)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    l = (max_c + min_c) / 2.0
    delta = max_c - min_c
    eps = 1e-6

    # Saturation
    s = np.where(l < 0.5, delta / (max_c + min_c + eps), delta / (2.0 - max_c - min_c + eps))

    # Vibrance weight: less saturated areas get more boost
    vib_weight = 1.0 - s

    s_adjusted = s * (1.0 + saturation) + vibrance * vib_weight
    s_adjusted = np.clip(s_adjusted, 0, 1)

    # Convert back to RGB (simplified: just interpolate to gray)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    blend = s_adjusted / (s + eps)
    blend = np.clip(blend, 0, 5)
    r = gray + (r - gray) * blend
    g = gray + (g - gray) * blend
    b = gray + (b - gray) * blend

    img = np.stack([r, g, b], axis=-1)
    img = np.clip(img, 0, 1)

    if was_int:
        img = (img * max_val).astype(np.uint16)
    return img


def _crop_rotate(
    img: np.ndarray,
    cx: float, cy: float, cw: float, ch: float,
    rotation_deg: float,
) -> np.ndarray:
    """Rotate (straighten / 90° orientation, expanding the canvas) THEN crop a 0-1 or
    uint16 image. Rotation happens first so the crop fractions are relative to the
    *rotated* frame — exactly what the web crop editor previews. Returns same dtype."""
    # Rotation via PIL — convert to float32 to avoid PIL uint16 issues. expand=True keeps
    # every pixel (grows the canvas, gray-filling the corners); the crop then trims them.
    if abs(rotation_deg) > 0.01:
        was_int = img.dtype != np.float32
        if was_int:
            img = img.astype(np.float32) / np.iinfo(np.uint16).max
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        rotated = pil_img.rotate(
            -rotation_deg, expand=True, resample=Image.BICUBIC, fillcolor=(128, 128, 128)
        )
        img = np.array(rotated).astype(np.float32) / 255.0
        if was_int:
            img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)

    # Crop (fractional coordinates of the rotated frame)
    h, w = img.shape[:2]
    x1 = max(0, int(cx * w))
    y1 = max(0, int(cy * h))
    x2 = min(w, int((cx + cw) * w))
    y2 = min(h, int((cy + ch) * h))
    if (x1, y1, x2, y2) != (0, 0, w, h):
        img = img[y1:y2, x1:x2]

    return img


def _decode_raw(raw_path: Path) -> np.ndarray | None:
    """RAW -> float32 gamma-encoded sRGB in [0,1], upright. None -> use darktable."""
    try:
        import rawpy
    except ImportError:
        return None
    try:
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                # Enable LibRaw's exposure normalization (stretch so the brightest
                # ~1% reaches white). Without it, RAWs render ~0.8 stop dark because
                # cameras expose to protect highlights — and the AI grades a bright
                # embedded-JPEG preview, so it never prescribes enough lift to compensate.
                no_auto_bright=False,
                gamma=(2.222, 4.5),
            )
    except Exception as exc:
        log.warning("rawpy failed for %s, trying darktable: %s", raw_path.name, exc)
        return None
    return rgb.astype(np.float32) / 65535.0


def _decode_jpeg(path: Path) -> np.ndarray:
    """Camera JPG -> the same buffer rawpy yields: float32 gamma-encoded sRGB [0,1],
    upright (EXIF orientation applied — PIL doesn't do it on its own like rawpy)."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            icc = im.info.get("icc_profile")
            if icc and im.mode in ("RGB", "CMYK"):
                try:
                    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                    if "srgb" not in (ImageCms.getProfileDescription(src) or "").lower():
                        im = ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"),
                                                       outputMode="RGB")
                except Exception as exc:  # broken profile: trust the pixels as sRGB
                    log.warning("%s: ignoring ICC profile (%s)", path.name, exc)
            rgb = np.asarray(im.convert("RGB"))
    except Exception as exc:
        raise DevelopError(f"Could not decode JPG {path.name}: {exc}") from exc
    return rgb.astype(np.float32) / 255.0


def develop(raw_path: Path, analysis_json: str, output_dir: Path) -> Path:
    """Run Stage 4 development. Returns path to the exported 16-bit TIFF."""
    analysis = json.loads(analysis_json)
    dp = analysis["develop"]

    jpeg = is_jpeg(raw_path)
    if jpeg:
        img = _decode_jpeg(raw_path)
    else:
        img = _decode_raw(raw_path)
        if img is None:
            return _develop_with_darktable(raw_path, dp, output_dir)

    # White balance. The RAW is already decoded with the camera's accurate as-shot
    # balance (use_camera_wb=True). Only apply an extra correction when the analysis
    # explicitly asks for a Kelvin override — otherwise we'd double-correct and skew
    # the color, which is the classic "colors look off" bug. A JPG's balance is baked
    # in by the camera, so a Kelvin override is skipped there.
    wb = dp.get("white_balance", {})
    if wb.get("mode") == "kelvin" and jpeg:
        log.info("%s: JPG source — skipping Kelvin white balance (as-shot is baked in)", raw_path.name)
    elif wb.get("mode") == "kelvin":
        kelvin = max(2000, min(15000, int(wb.get("temp", 6500))))
        kr, kg, kb = _wb_relative_multipliers(kelvin)
        kr, kg, kb = _apply_tint(kr, kg, kb, wb.get("tint", 0))
        img[..., 0] *= kr
        img[..., 1] *= kg
        img[..., 2] *= kb

    # Exposure — apply in LINEAR light so +1 EV == 2x scene luminance. The buffer here is
    # gamma-encoded (postprocess gamma≈2.222), and multiplying an encoded buffer directly
    # would over-brighten (~2.2x per stop) and crush the toe; round-trip through the gamma
    # around the multiply so a stop is a true stop.
    ev = dp.get("exposure_ev", 0.0)
    if abs(ev) > 1e-4:
        g = 2.222
        lin = np.clip(img, 0.0, None) ** g
        lin *= 2.0 ** ev
        img = lin ** (1.0 / g)

    # Highlight rolloff instead of a hard clip: bright skin/specular that would blow out to
    # flat white (common when auto-brighten over-lifts a low-key scene and the AI adds a bit
    # of exposure on top) is compressed smoothly toward 1.0, keeping tone/detail in highlights.
    img = _highlight_rolloff(img, CONFIG.highlight_rolloff_knee)

    # Contrast / highlights / shadows
    img = _tone_curve(img, dp.get("contrast", 0.0), dp.get("highlights", 0.0), dp.get("shadows", 0.0))

    # Saturation / vibrance
    img = _apply_saturation_vibrance(img, dp.get("saturation", 0.0), dp.get("vibrance", 0.0))

    # Final highlight rolloff — the contrast S-curve above pushes highlights back up, so we
    # protect them again as the last step before clipping.
    img = _highlight_rolloff(img, CONFIG.highlight_rolloff_knee)

    # Convert back to 16-bit
    img_16 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)

    # Crop and rotate
    crop = dp.get("crop", {})
    img_16 = _crop_rotate(
        img_16,
        crop.get("x", 0.0), crop.get("y", 0.0),
        crop.get("w", 1.0), crop.get("h", 1.0),
        dp.get("rotation_deg", 0.0),
    )

    dest = output_dir / f"{raw_path.stem}.tiff"
    tifffile.imwrite(
        str(dest), img_16,
        compression=CONFIG.develop_tiff_compression,
    )
    log.info("Developed %s -> %s (%dx%d)", raw_path.name, dest.name, img_16.shape[1], img_16.shape[0])
    return dest


def _develop_with_darktable(raw_path: Path, dp: dict, output_dir: Path) -> Path:
    """Fallback: use darktable-cli with minimal settings to export a TIFF."""
    darktable = find_tool("darktable-cli")
    if darktable is None:
        raise DevelopError(f"No RAW decoder available for {raw_path.name}")

    dest = output_dir / f"{raw_path.stem}.tiff"
    result = subprocess.run(
        [
            darktable, str(raw_path), str(dest),
            "--width", "4096", "--height", "4096",
            "--core", "--library", ":memory:",
            "--bpp", str(CONFIG.develop_output_bps),
        ],
        capture_output=True, timeout=300,
    )
    if result.returncode != 0 or not dest.exists():
        raise DevelopError(
            f"darktable-cli failed for {raw_path.name}: {result.stderr.decode(errors='replace')[-500:]}"
        )
    log.info("Developed (darktable) %s -> %s", raw_path.name, dest.name)
    return dest
