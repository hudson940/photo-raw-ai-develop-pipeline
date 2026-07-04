"""Stages 5-6: deterministic retouch engine (OpenCV) + optional generative background.

The previous implementation pushed the whole frame through Stable Diffusion
img2img, which *regenerates* the photo: faces shift, texture is invented,
and resolution drops to 1024px. Real retouching is a set of small, masked,
reversible operations, so this module performs them directly on the 16-bit
TIFF at full resolution:

- skin softening    frequency separation: skin tone is evened out while
                    pores and fine texture are preserved
- blemish softening high-frequency outlier suppression inside the skin mask
- eye brightening   feathered midtone lift at the detected eye landmarks
- teeth whitening   yellow-cast reduction inside the detected mouth region
- clothing contrast CLAHE local contrast on the subject, skin excluded
- background        keep / blur / studio backdrop / generative replace

Face detection is OpenCV YuNet (bundled ONNX model, CPU, milliseconds).
Subject/background separation is rembg (U²-Net). Generative background
replacement is the only step that still uses ComfyUI, and it falls back to
the deterministic 'studio' backdrop when ComfyUI is unreachable.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

from .config import CONFIG

log = logging.getLogger("retouch")

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_YUNET_PATH = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"

# Overall strength multiplier per requested intensity level
_INTENSITY = {"subtle": 0.6, "natural": 1.0, "polished": 1.3}

_rembg_session = None  # lazy singleton; loading the U²-Net weights takes seconds


class RetouchError(Exception):
    pass


@dataclass
class Face:
    x: float
    y: float
    w: float
    h: float
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_left: tuple[float, float]


# --------------------------------------------------------------------------- I/O

def _load_image(path: Path) -> np.ndarray:
    """Load a TIFF/JPEG/PNG as float32 RGB in [0, 1]."""
    if path.suffix.lower() in (".tif", ".tiff"):
        img = tifffile.imread(str(path))
    else:
        img = np.array(Image.open(path).convert("RGB"))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3].astype(np.float32)
    if img.max() > 1.0:
        img /= 65535.0 if img.max() > 255 else 255.0
    return np.ascontiguousarray(img)


def _save_tiff(img: np.ndarray, path: Path) -> None:
    out = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(str(path), out, compression=CONFIG.develop_tiff_compression)


# ------------------------------------------------------------------ face detection

def _detect_faces(img: np.ndarray) -> list[Face]:
    """Run YuNet across a small scale pyramid; return boxes/landmarks in full-res coords.

    YuNet misses faces that are very large relative to the frame (tight
    close-ups), so if nothing is found at 1024px we retry at smaller input
    sizes where the face occupies fewer pixels.
    """
    if not _YUNET_PATH.exists():
        log.warning("YuNet model missing at %s — face-dependent retouch disabled", _YUNET_PATH)
        return []

    h, w = img.shape[:2]
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    for target in (1024, 640, 320):
        scale = min(1.0, target / max(h, w))
        bgr = cv2.cvtColor(
            cv2.resize(img_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2BGR,
        )
        detector = cv2.FaceDetectorYN.create(str(_YUNET_PATH), "", (bgr.shape[1], bgr.shape[0]),
                                             score_threshold=0.7)
        _, dets = detector.detect(bgr)
        if dets is None or len(dets) == 0:
            continue
        faces: list[Face] = []
        for d in dets:
            v = d / scale  # back to full-res coordinates (score at [14] becomes garbage; unused)
            faces.append(Face(
                x=v[0], y=v[1], w=v[2], h=v[3],
                right_eye=(v[4], v[5]), left_eye=(v[6], v[7]),
                mouth_right=(v[10], v[11]), mouth_left=(v[12], v[13]),
            ))
        return faces
    return []


# ------------------------------------------------------------------------- masks

def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, sigma))


def _skin_mask(img: np.ndarray, faces: list[Face]) -> np.ndarray:
    """Float mask of likely skin: face/neck geometry gated by per-face skin color."""
    h, w = img.shape[:2]
    region = np.zeros((h, w), np.uint8)
    color_gate = np.zeros((h, w), np.float32)

    ycrcb = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[..., 1].astype(np.float32)
    cb = ycrcb[..., 2].astype(np.float32)

    for f in faces:
        cx, cy = f.x + f.w / 2, f.y + f.h / 2
        # face oval (forehead to chin) + neck/chest blob below
        cv2.ellipse(region, (int(cx), int(cy)), (int(f.w * 0.68), int(f.h * 0.85)),
                    0, 0, 360, 255, -1)
        cv2.ellipse(region, (int(cx), int(f.y + f.h * 1.45)), (int(f.w * 0.60), int(f.h * 0.65)),
                    0, 0, 360, 255, -1)

        # adaptive skin color from the face core (robust across skin tones)
        x1, x2 = int(cx - f.w * 0.22), int(cx + f.w * 0.22)
        y1, y2 = int(cy - f.h * 0.18), int(cy + f.h * 0.22)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cr_m, cr_s = cr[y1:y2, x1:x2].mean(), cr[y1:y2, x1:x2].std() + 2.0
        cb_m, cb_s = cb[y1:y2, x1:x2].mean(), cb[y1:y2, x1:x2].std() + 2.0
        dist = ((cr - cr_m) / (3.0 * cr_s)) ** 2 + ((cb - cb_m) / (3.0 * cb_s)) ** 2
        color_gate = np.maximum(color_gate, np.clip(1.5 - dist, 0, 1))

        # exclude eyes and mouth — smoothing there kills the portrait
        eye_r = int(f.w * 0.13)
        for ex, ey in (f.right_eye, f.left_eye):
            cv2.circle(region, (int(ex), int(ey)), eye_r, 0, -1)
        mx = (f.mouth_left[0] + f.mouth_right[0]) / 2
        my = (f.mouth_left[1] + f.mouth_right[1]) / 2
        mouth_w = np.hypot(f.mouth_right[0] - f.mouth_left[0],
                           f.mouth_right[1] - f.mouth_left[1])
        cv2.ellipse(region, (int(mx), int(my)), (int(mouth_w * 0.75), int(mouth_w * 0.45)),
                    0, 0, 360, 0, -1)

    if not faces:
        return np.zeros((h, w), np.float32)

    face_w = max(f.w for f in faces)
    mask = (region.astype(np.float32) / 255.0) * np.clip(color_gate, 0, 1)
    return np.clip(_feather(mask, face_w * 0.04), 0, 1)


def _subject_alpha(img: np.ndarray) -> np.ndarray | None:
    """Subject (person) alpha matte from rembg, full-res float in [0, 1]."""
    global _rembg_session
    try:
        from rembg import new_session, remove
    except ImportError:
        log.warning("rembg not installed — background/clothing ops skipped")
        return None
    try:
        if _rembg_session is None:
            _rembg_session = new_session(CONFIG.rembg_model)
        h, w = img.shape[:2]
        scale = min(1.0, 1600 / max(h, w))
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8))
        cut = remove(pil, session=_rembg_session)
        alpha = np.array(cut.split()[-1]).astype(np.float32) / 255.0
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(_feather(alpha, max(h, w) * 0.0015), 0, 1)
    except Exception as exc:
        log.warning("rembg segmentation failed: %s", exc)
        return None


# --------------------------------------------------------------------- operations

def _soften_skin(img: np.ndarray, skin: np.ndarray, strength: float,
                 face_w: float) -> np.ndarray:
    """Frequency separation: even out skin tone on the low frequency, keep texture.

    detail (pores, hair) = img - base; the base is smoothed with an
    edge-preserving bilateral filter, then detail is added back untouched,
    so the result never looks 'plastic' — natural pores are preserved.
    """
    if strength <= 0.01 or skin.max() < 0.01:
        return img

    sigma = max(2.0, face_w * 0.016)
    base = cv2.GaussianBlur(img, (0, 0), sigma)
    detail = img - base

    # bilateral on a downscaled base (base has no fine detail, nothing is lost)
    h, w = img.shape[:2]
    scale = min(1.0, 1400 / max(h, w))
    small = cv2.resize(base, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 9, 0.08, 8)
    smooth_base = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    softened = smooth_base + detail
    blend = (skin * min(max(strength, 0.0), 0.85))[..., None]
    return img * (1 - blend) + softened * blend


def _remove_blemishes(img: np.ndarray, skin: np.ndarray, faces: list[Face],
                      strength: float) -> np.ndarray:
    """Erase pimples, acne marks, skin tags and stray hairs *over skin* by cloning
    neighbouring skin over them (cv2.inpaint — the deterministic clone/heal brush).

    Detection: a morphological black-hat finds small dark features on lighter skin
    (spots, thin stray hairs); an excess-redness test catches inflamed pimples. Both
    are gated by the color-matched skin mask, so facial features (brows, nostrils,
    lips, eyes — none skin-colored) are never touched.
    """
    if strength <= 0.01 or not faces or skin.max() < 0.01:
        return img
    h, w = img.shape[:2]
    face_w = max(f.w for f in faces)

    # detect on a downscaled copy for speed; blemish scale stays sensible
    scale = min(1.0, 1600 / max(h, w))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    fw_s = face_w * scale
    small = cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
    skin_s = cv2.resize(skin, (sw, sh), interpolation=cv2.INTER_LINEAR)

    k = int(max(3, fw_s * 0.03)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab).astype(np.float32)
    redness = np.clip(lab[..., 1] - cv2.GaussianBlur(lab[..., 1], (0, 0), k), 0, None)

    # high base threshold so only strong, isolated features register (not skin texture)
    thr = 24.0 - 10.0 * min(strength, 1.0)
    raw = (((blackhat > thr) | (redness > 16.0)) & (skin_s > 0.4)).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Keep only genuine blemishes: small compact blobs (pimples, marks, skin tags) or
    # thin short strands (stray hairs). Reject wrinkles/folds, which form large or
    # elongated components — this is what stops the detector eating textured skin.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = np.zeros_like(raw)
    max_blob = (fw_s * 0.05) ** 2   # bigger than ~5% of face width is not a blemish
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 2 or area > max_blob:
            continue
        extent = area / max(1, bw * bh)               # how filled its bounding box is
        elong = max(bw, bh) / max(1, min(bw, bh))     # aspect ratio
        is_spot = extent > 0.4 and elong < 2.5
        is_hair = elong >= 3.0 and min(bw, bh) <= max(2, fw_s * 0.008) and max(bw, bh) < fw_s * 0.4
        if is_spot or is_hair:
            keep[labels == i] = 1

    # safety valve: if we somehow selected a large fraction of skin, the detector is
    # misfiring on texture — do nothing rather than smear the face.
    skin_px = max(1.0, float((skin_s > 0.4).sum()))
    if keep.sum() == 0 or keep.sum() > skin_px * 0.05:
        if keep.sum() > 0:
            log.info("Blemish detector over-triggered (%.0f%% of skin) — skipping to stay safe",
                     100 * keep.sum() / skin_px)
        return img

    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    full_mask = cv2.resize(keep * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    full_mask = (full_mask * (skin > 0.4)).astype(np.uint8)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    inpainted = cv2.inpaint(u8, full_mask, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0

    m = (_feather(full_mask.astype(np.float32) / 255.0, face_w * 0.006) * skin)[..., None]
    m = np.clip(m * min(1.0, strength + 0.1), 0, 1)
    return img * (1 - m) + inpainted * m


def _brighten_eyes(img: np.ndarray, faces: list[Face], strength: float) -> np.ndarray:
    """Brighten the whites of the eyes (sclera) and take down bloodshot redness.

    Gated to bright, low-saturation pixels inside the eye disk, so the iris and
    lashes are left alone.
    """
    if strength <= 0.01 or not faces:
        return img
    h, w = img.shape[:2]
    face_w = max(f.w for f in faces)
    mask = np.zeros((h, w), np.float32)
    for f in faces:
        r = max(3, int(f.w * 0.11))
        for ex, ey in (f.right_eye, f.left_eye):
            cv2.circle(mask, (int(ex), int(ey)), r, 1.0, -1)
    mask = _feather(mask, face_w * 0.02)

    fimg = np.clip(img, 0, 1).astype(np.float32)
    hsv = cv2.cvtColor(fimg, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    sclera = mask * np.clip((val - 0.35) / 0.4, 0, 1) * np.clip((0.5 - sat) / 0.4, 0, 1)
    s = min(max(strength, 0.0), 0.8)

    lab = cv2.cvtColor(fimg, cv2.COLOR_RGB2Lab)
    lab[..., 0] = np.clip(lab[..., 0] + sclera * s * 9.0, 0, 100)   # brighten whites
    lab[..., 1] = lab[..., 1] - sclera * s * 4.0                    # less bloodshot red
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _iris_pop(img: np.ndarray, faces: list[Face], strength: float) -> np.ndarray:
    """Add a subtle saturation/clarity pop to the irises (not the pupil or sclera)."""
    if strength <= 0.01 or not faces:
        return img
    h, w = img.shape[:2]
    face_w = max(f.w for f in faces)
    mask = np.zeros((h, w), np.float32)
    for f in faces:
        r = max(2, int(f.w * 0.055))
        for ex, ey in (f.right_eye, f.left_eye):
            cv2.circle(mask, (int(ex), int(ey)), r, 1.0, -1)
    mask = _feather(mask, face_w * 0.012)

    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    # iris = mid-tone; exclude the dark pupil and the bright sclera
    iris = mask * np.clip((val - 0.12) / 0.2, 0, 1) * np.clip((0.9 - val) / 0.3, 0, 1)
    s = min(max(strength, 0.0), 0.6)
    hsv[..., 1] = np.clip(sat + iris * s * 0.22, 0, 1)             # saturation pop
    hsv[..., 2] = np.clip(val + iris * s * (0.5 - val) * 0.12, 0, 1)  # gentle local contrast
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)


def _reduce_dark_circles(img: np.ndarray, faces: list[Face], strength: float,
                         skin: np.ndarray) -> np.ndarray:
    """Lighten under-eye dark circles / eye bags toward the surrounding skin brightness."""
    if strength <= 0.01 or not faces:
        return img
    h, w = img.shape[:2]
    face_w = max(f.w for f in faces)
    mask = np.zeros((h, w), np.float32)
    for f in faces:
        for ex, ey in (f.right_eye, f.left_eye):
            cv2.ellipse(mask, (int(ex), int(ey + f.h * 0.11)),
                        (int(f.w * 0.17), int(f.h * 0.09)), 0, 0, 360, 1.0, -1)
    mask = _feather(mask, face_w * 0.03)
    if skin.max() > 0:
        mask = mask * skin
    if mask.max() < 0.01:
        return img

    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    l = lab[..., 0]
    l_base = cv2.GaussianBlur(l, (0, 0), face_w * 0.05)   # local skin brightness
    lift = np.clip(l_base - l, 0, None)                   # positive where darker than skin
    s = min(max(strength, 0.0), 0.85)
    lab[..., 0] = np.clip(l + mask * lift * 0.75 * s, 0, 100)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _reduce_dewlap(img: np.ndarray, faces: list[Face], strength: float) -> np.ndarray:
    """Subtly reduce a dewlap / double chin by nudging the under-chin skin upward.

    A gentle localized 'liquify push': a smooth displacement field pulls the sagging
    skin below the chin up toward the jawline. The pull is zero at the chin (a smooth
    ramp, no seam), peaks just below it, and fades out sideways and downward — capped
    at ~7% of face height so it never looks stretched or fake.
    """
    if strength <= 0.01 or not faces:
        return img
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    disp = np.zeros((h, w), np.float32)
    for f in faces:
        cx = (f.mouth_left[0] + f.mouth_right[0]) / 2.0
        chin_y = f.y + f.h                     # ~ bottom of the face box (the chin)
        center_y = chin_y + f.h * 0.14         # push strongest a little below the chin
        sx, sy = f.w * 0.45, f.h * 0.20
        amp = f.h * 0.07 * min(max(strength, 0.0), 1.0)
        gx = np.exp(-((map_x - cx) ** 2) / (2.0 * sx * sx))
        gy = np.exp(-((map_y - center_y) ** 2) / (2.0 * sy * sy))
        ramp = np.clip((map_y - chin_y) / (f.h * 0.06), 0.0, 1.0)   # 0 at chin, smooth
        disp = np.maximum(disp, amp * gx * gy * ramp)
    map_y2 = map_y + disp                       # sample from below -> pulls skin up
    return cv2.remap(img, map_x, map_y2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _skin_tone_correction(img: np.ndarray, skin: np.ndarray, strength: float) -> np.ndarray:
    """Even out blotchy skin chroma and give a healthy tone, keeping luminance texture."""
    if strength <= 0.01 or skin.max() < 0.01:
        return img
    h, w = img.shape[:2]
    s = min(max(strength, 0.0), 0.8)
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    a, b = lab[..., 1], lab[..., 2]

    sigma = max(h, w) * 0.005
    a_s = cv2.GaussianBlur(a, (0, 0), sigma)
    b_s = cv2.GaussianBlur(b, (0, 0), sigma)
    even = skin * s * 0.6
    lab[..., 1] = a * (1 - even) + a_s * even     # smooth out red/green blotches
    lab[..., 2] = b * (1 - even) + b_s * even

    warm = skin * s
    lab[..., 1] = lab[..., 1] + warm * 0.8        # healthy warmth (away from sallow/green)
    lab[..., 2] = lab[..., 2] + warm * 0.6
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _whiten_teeth(img: np.ndarray, faces: list[Face], strength: float) -> np.ndarray:
    if strength <= 0.01 or not faces:
        return img
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for f in faces:
        mx = (f.mouth_left[0] + f.mouth_right[0]) / 2
        my = (f.mouth_left[1] + f.mouth_right[1]) / 2
        mw = np.hypot(f.mouth_right[0] - f.mouth_left[0], f.mouth_right[1] - f.mouth_left[1])
        if mw < 4:
            continue
        canvas = np.zeros((h, w), np.uint8)
        cv2.ellipse(canvas, (int(mx), int(my)), (int(mw * 0.6), int(mw * 0.35)), 0, 0, 360, 255, -1)
        mask = np.maximum(mask, _feather(canvas.astype(np.float32) / 255.0, mw * 0.08))
    if mask.max() < 0.01:
        return img

    lab = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2Lab)
    l_ch, b_ch = lab[..., 0], lab[..., 2]
    # gate to bright, yellowish pixels inside the mouth region = teeth, not lips
    gate = mask * np.clip((l_ch - 40.0) / 25.0, 0, 1) * np.clip((b_ch - 2.0) / 10.0, 0, 1)
    s = min(max(strength, 0.0), 0.8)
    lab[..., 2] -= gate * s * 18.0   # pull yellow toward neutral
    lab[..., 0] += gate * s * 6.0    # slight brightening
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _clothing_contrast(img: np.ndarray, subject: np.ndarray | None,
                       skin: np.ndarray, strength: float) -> np.ndarray:
    """Subtle CLAHE 'clarity' on the subject's clothing (subject minus skin)."""
    if strength <= 0.01:
        return img
    lab = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2Lab)
    l_u16 = (lab[..., 0] / 100.0 * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_u16).astype(np.float32) / 65535.0 * 100.0

    cloth = (subject if subject is not None else np.ones(skin.shape, np.float32)) * (1 - skin)
    m = cloth * min(max(strength, 0.0), 0.7)
    lab[..., 0] = lab[..., 0] * (1 - m) + l_eq * m
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


# -------------------------------------------------------------------- backgrounds

def _normalized_blur(img: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur only the background. Normalized convolution weights out the
    subject so its colors never bleed into the background (no halo at the edge)."""
    inv = (1 - alpha)[..., None]
    blurred = cv2.GaussianBlur(img * inv, (0, 0), sigma)
    norm = cv2.GaussianBlur(1 - alpha, (0, 0), sigma)[..., None]
    return blurred / np.maximum(norm, 1e-4)


def _blur_background(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Soft bokeh-style background for busy/distracting scenes."""
    h, w = img.shape[:2]
    bg = _normalized_blur(img, alpha, max(h, w) * 0.01)
    bg = np.clip(bg * 0.96, 0, 1)  # slight darkening pushes attention to the subject
    a = alpha[..., None]
    return img * a + bg * (1 - a)


def _smooth_background(img: np.ndarray, alpha: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """De-wrinkle a photographic backdrop: clone out imperfections, then smooth creases.

    For studio/paper/cloth backdrops that show folds, creases, lint or marks. Unlike
    'blur' (which turns the background into soft bokeh) this keeps the backdrop reading
    as a flat backdrop — it removes the *texture* while preserving the overall lighting:

      1. isolate the background with the subject alpha matte (1 - alpha)
      2. clone out hard imperfections (spots, lint, seams) with inpainting, the
         deterministic equivalent of a clone/heal brush
      3. smooth the remaining soft creases with an edge-aware background-only blur
    """
    h, w = img.shape[:2]
    inv = 1.0 - alpha
    bg_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # 1-2. detect imperfections as local deviations from a median-smoothed backdrop
    #      (fast on a downscaled copy), then clone neighbouring backdrop over them.
    scale = min(1.0, 1400 / max(h, w))
    small = cv2.resize(bg_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_inv = cv2.resize(inv, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_LINEAR)
    med = cv2.medianBlur(small, 5)
    dev = np.abs(small.astype(np.float32) - med).mean(axis=2)
    spots = ((dev > 10.0) & (small_inv > 0.6)).astype(np.uint8) * 255
    spots = cv2.dilate(spots, np.ones((3, 3), np.uint8))
    spots = cv2.resize(spots, (w, h), interpolation=cv2.INTER_NEAREST)
    spots = (spots * (inv > 0.6)).astype(np.uint8)  # never touch the subject
    cleaned = cv2.inpaint(bg_u8, spots, 4, cv2.INPAINT_TELEA).astype(np.float32) / 255.0

    # 3. smooth soft creases; sigma scales with the requested strength
    sigma = max(h, w) * 0.006 * max(0.3, min(2.0, strength))
    smoothed = _normalized_blur(cleaned, alpha, sigma)

    a = alpha[..., None]
    return np.clip(img * a + smoothed * (1 - a), 0, 1)


def _studio_background(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Replace the background with a neutral studio-gray gradient."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 0.86 - 0.22 * (yy / h)                       # brighter top, darker bottom
    r = np.hypot((xx - w / 2) / w, (yy - h * 0.4) / h)  # gentle vignette
    grad = np.clip(grad - 0.10 * r, 0, 1)
    bg = np.stack([grad, grad, grad * 1.005], axis=-1)  # hair-width cool cast
    a = alpha[..., None]
    return np.clip(img * a + bg * (1 - a), 0, 1)


def _replace_background_comfy(img: np.ndarray, alpha: np.ndarray, prompt: str) -> np.ndarray:
    """Generative background via ComfyUI inpainting. Only the background region is
    regenerated; the full-resolution subject is composited back over the result."""
    import httpx

    workflow = json.loads((_WORKFLOW_DIR / "inpaint.json").read_text())["nodes"]

    h, w = img.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    sw, sh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_alpha = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_LINEAR)

    tmp = CONFIG.output / f".comfy_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(exist_ok=True)
    img_path = tmp / "input.png"
    mask_path = tmp / "mask.png"
    Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8)).save(img_path)
    # mask: white = regenerate (background), slightly grown into the subject edge
    bg_mask = np.clip((1 - small_alpha) * 255, 0, 255).astype(np.uint8)
    bg_mask = cv2.dilate(bg_mask, np.ones((5, 5), np.uint8))
    Image.fromarray(np.stack([bg_mask] * 3, axis=-1)).save(mask_path)

    with httpx.Client() as client:
        base = CONFIG.comfyui_url
        try:
            client.get(f"{base}/queue", timeout=5).raise_for_status()
        except Exception as exc:
            raise RetouchError(f"ComfyUI not reachable at {base}: {exc}")

        def upload(p: Path) -> str:
            with open(p, "rb") as fh:
                resp = client.post(f"{base}/upload/image",
                                   files={"image": (p.name, fh, "image/png")}, timeout=30)
            resp.raise_for_status()
            return resp.json().get("name", p.name)

        workflow["1"]["inputs"]["image"] = upload(img_path)
        workflow["2"]["inputs"]["image"] = upload(mask_path)
        workflow["6"]["inputs"]["denoise"] = 1.0
        workflow["6"]["inputs"]["seed"] = int(time.time())
        workflow["7"]["inputs"]["ckpt_name"] = CONFIG.comfyui_checkpoint
        workflow["8"]["inputs"]["text"] = f"{prompt}, photorealistic, natural lighting, high quality photo"

        resp = client.post(f"{base}/prompt",
                           json={"prompt": workflow, "client_id": f"pipeline-{uuid.uuid4().hex[:8]}"},
                           timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        log.info("Background generation submitted (prompt_id=%s)", prompt_id)

        deadline = time.time() + CONFIG.comfyui_timeout
        history = None
        while time.time() < deadline:
            data = client.get(f"{base}/history/{prompt_id}", timeout=10).json()
            if prompt_id in data:
                history = data[prompt_id]
                break
            time.sleep(2)
        if history is None:
            raise RetouchError(f"Timed out waiting for ComfyUI job {prompt_id}")

        filename = None
        for node_output in history.get("outputs", {}).values():
            if node_output.get("images"):
                filename = node_output["images"][0]["filename"]
                break
        if filename is None:
            raise RetouchError("ComfyUI returned no output image")

        resp = client.get(f"{base}/view",
                          params={"filename": filename, "subfolder": "", "type": "output"},
                          timeout=60)
        resp.raise_for_status()
        result_path = tmp / "result.png"
        result_path.write_bytes(resp.content)

    generated = _load_image(result_path)
    bg_full = cv2.resize(generated, (w, h), interpolation=cv2.INTER_LANCZOS4)
    a = alpha[..., None]
    return np.clip(img * a + bg_full * (1 - a), 0, 1)


def _apply_background(img: np.ndarray, action: str, prompt: str,
                      alpha: np.ndarray | None) -> np.ndarray:
    if action == "keep":
        return img
    if alpha is None or alpha.max() < 0.05 or alpha.mean() > 0.95:
        log.warning("No usable subject mask — leaving background unchanged")
        return img
    if action == "blur":
        return _blur_background(img, alpha)
    if action == "smooth":
        return _smooth_background(img, alpha)
    if action == "studio":
        return _studio_background(img, alpha)
    if action == "replace":
        try:
            return _replace_background_comfy(img, alpha, prompt or "clean neutral studio backdrop")
        except RetouchError as exc:
            log.warning("Generative replace failed (%s) — using studio backdrop instead", exc)
            return _studio_background(img, alpha)
    log.warning("Unknown background action %r — keeping original", action)
    return img


# -------------------------------------------------------------------- entry point

def retouch(tiff_path: Path, analysis_json: str) -> Path:
    """Run Stages 5-6. Returns the retouched image path, or the input path
    unchanged when there is nothing to do."""
    analysis = json.loads(analysis_json)
    rp = analysis.get("retouch", {})
    bg = rp.get("background", {}) or {}
    bg_action = bg.get("action", "keep")

    wants_face_work = rp.get("is_portrait", False) and any((
        rp.get("skin_smoothing", 0) > 0.01,
        rp.get("remove_blemishes", False),
        rp.get("brighten_eyes", 0) > 0.01,
        rp.get("whiten_teeth", 0) > 0.01,
        rp.get("iris_enhance", 0) > 0.01,
        rp.get("reduce_dark_circles", 0) > 0.01,
        rp.get("reduce_dewlap", 0) > 0.01,
        rp.get("skin_tone_correction", 0) > 0.01,
    ))
    wants_cloth = rp.get("clothing_contrast", 0) > 0.01
    wants_bg = bg_action != "keep"

    if CONFIG.skip_retouch or not (wants_face_work or wants_cloth or wants_bg):
        log.info("Skipping retouch (nothing requested, skip=%s)", CONFIG.skip_retouch)
        return tiff_path

    mult = _INTENSITY.get(rp.get("intensity", "natural"), 1.0)
    img = _load_image(tiff_path)

    faces = _detect_faces(img) if wants_face_work else []
    if wants_face_work and not faces:
        log.warning("Analysis says portrait but no face detected — skipping face retouch")

    skin = _skin_mask(img, faces) if faces else np.zeros(img.shape[:2], np.float32)
    alpha = _subject_alpha(img) if (wants_bg or wants_cloth) else None

    if faces:
        face_w = max(f.w for f in faces)
        # geometry first (a warp changes where everything is), then tone/color ops
        img = _reduce_dewlap(img, faces, rp.get("reduce_dewlap", 0) * mult)
        # blemish erase next, so skin softening blends over the cloned patches
        if rp.get("remove_blemishes", False):
            img = _remove_blemishes(img, skin, faces, 0.8 * mult)
        img = _soften_skin(img, skin, rp.get("skin_smoothing", 0) * mult, face_w)
        img = _skin_tone_correction(img, skin, rp.get("skin_tone_correction", 0) * mult)
        img = _reduce_dark_circles(img, faces, rp.get("reduce_dark_circles", 0) * mult, skin)
        img = _brighten_eyes(img, faces, rp.get("brighten_eyes", 0) * mult)
        img = _iris_pop(img, faces, rp.get("iris_enhance", 0) * mult)
        img = _whiten_teeth(img, faces, rp.get("whiten_teeth", 0) * mult)

    img = _clothing_contrast(img, alpha, skin, rp.get("clothing_contrast", 0) * mult)
    img = _apply_background(img, bg_action, bg.get("replace_prompt", ""), alpha)

    dest = Path(tiff_path).with_suffix(".retouched.tiff")
    _save_tiff(img, dest)
    log.info("Retouched %s -> %s (faces=%d, bg=%s)", tiff_path.name, dest.name,
             len(faces), bg_action)
    return dest
