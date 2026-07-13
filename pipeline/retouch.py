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

def dump_skin_mask(tiff_path: Path, out_path: Path, long_edge: int = 1500) -> tuple[Path, float]:
    """Save a side-by-side [original | skin-selection overlay] JPEG so the mask can be inspected.

    The overlay tints selected skin red with brightness proportional to the mask value, so partly
    selected (feathered) zones read as darker red and holes stay their original color.
    """
    img = _load_image(Path(tiff_path))
    faces = _detect_faces(img)
    alpha = _subject_alpha(img) if faces else None
    skin = _skin_mask(img, faces, alpha) if faces else np.zeros(img.shape[:2], np.float32)

    over = img.copy()
    m = np.clip(skin, 0, 1)
    over[..., 0] = np.clip(img[..., 0] * (1 - m) + (0.25 * img[..., 0] + 0.75) * m, 0, 1)
    over[..., 1] = img[..., 1] * (1 - 0.55 * m)
    over[..., 2] = img[..., 2] * (1 - 0.55 * m)
    side = np.concatenate([img, over], axis=1)

    h, w = side.shape[:2]
    scale = min(1.0, long_edge * 2 / w)          # long_edge per panel
    if scale < 1.0:
        side = cv2.resize(side, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).save(out_path, quality=90)
    return out_path, float((skin > 0.5).mean())


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


def _skin_mask(img: np.ndarray, faces: list[Face],
               subject: np.ndarray | None = None) -> np.ndarray:
    """Float mask of ALL likely skin — face/neck + body, eyes and lips INCLUDED.

    The face/neck comes from geometry gated by per-face skin color. When a subject matte is
    supplied, skin-colored pixels ELSEWHERE on the subject — arms, shoulders, chest — are added
    too, so body skin is tone-corrected to match the face. Gating by the subject keeps skin-toned
    background out. Eyes/lips are NOT excluded here (so color correction reaches all skin evenly);
    ops that must not touch features (softening, blemish removal) multiply by `_feature_exclusion`.
    """
    h, w = img.shape[:2]
    region = np.zeros((h, w), np.uint8)
    color_gate = np.zeros((h, w), np.float32)  # per-pixel skin-color probability (whole frame)

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
        # a slightly looser gate (3.3σ, offset 1.6) so the nose, perioral skin and lit/shadowed
        # areas — which drift from the exact face-core color — are still selected, not left as
        # uncorrected (green) holes. The face geometry keeps this from spilling off the face.
        dist = ((cr - cr_m) / (3.3 * cr_s)) ** 2 + ((cb - cb_m) / (3.3 * cb_s)) ** 2
        color_gate = np.maximum(color_gate, np.clip(1.6 - dist, 0, 1))

    if not faces:
        return np.zeros((h, w), np.float32)

    face_w = max(f.w for f in faces)
    mask = (region.astype(np.float32) / 255.0) * np.clip(color_gate, 0, 1)

    # fill small color holes inside the face geometry (nose specular, skin next to the lips) so
    # the whole face is corrected — not only pixels that matched the face-core color exactly.
    # The close is bounded by the geometric region, so it can't grow past the face/neck.
    geo = (region > 0)
    k = max(3, int(face_w * 0.05)) | 1
    binm = cv2.morphologyEx((mask > 0.35).astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = np.maximum(mask, binm.astype(np.float32) * geo)

    if subject is not None:
        # body skin: skin-colored pixels on the subject (arms/shoulders/chest). Require higher
        # color confidence than the face path so skin-toned fabric isn't swept in, and REJECT HAIR
        # — hair is close to skin color but noticeably LESS saturated (lower Lab chroma) and more
        # textured (strands) than skin, so gate the body path on both. (The face oval is left at
        # full strength; only this whole-subject path risks catching the long hair on shoulders.)
        lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        tex = cv2.GaussianBlur(np.abs(g - cv2.GaussianBlur(g, (0, 0), max(h, w) * 0.002)),
                               (0, 0), max(h, w) * 0.004)
        not_hair = np.clip((chroma - 5.0) / 7.0, 0, 1) * np.clip((20.0 - tex) / 12.0, 0, 1)
        body = np.clip((color_gate - 0.25) / 0.75, 0, 1) * np.clip(subject, 0, 1) * not_hair
        mask = np.maximum(mask, body)

    return np.clip(_feather(mask, face_w * 0.04), 0, 1)


def _feature_exclusion(shape: tuple, faces: list[Face], face_w: float) -> np.ndarray:
    """Feathered mask (1 = exclude) over the eyes and lips. Multiplied into the skin mask for ops
    that must NOT touch those features — skin softening and blemish removal (blurring/cloning the
    eyes or lips would ruin the portrait). Colour tone correction deliberately does NOT use this,
    so skin colour is corrected evenly right up to and across the eyes and mouth."""
    h, w = shape[:2]
    excl = np.zeros((h, w), np.float32)
    for f in faces:
        eye_r = int(f.w * 0.12)
        for ex, ey in (f.right_eye, f.left_eye):
            cv2.circle(excl, (int(ex), int(ey)), eye_r, 1.0, -1)
        mx = (f.mouth_left[0] + f.mouth_right[0]) / 2
        my = (f.mouth_left[1] + f.mouth_right[1]) / 2
        mouth_w = np.hypot(f.mouth_right[0] - f.mouth_left[0], f.mouth_right[1] - f.mouth_left[1])
        cv2.ellipse(excl, (int(mx), int(my)), (int(mouth_w * 0.62), int(mouth_w * 0.34)),
                    0, 0, 360, 1.0, -1)
    return _feather(np.clip(excl, 0, 1), face_w * 0.012)


def _solidify_alpha(alpha: np.ndarray) -> np.ndarray:
    """Make the subject interior fully opaque, keeping a soft edge only at the true boundary.

    rembg often returns a semi-transparent matte over light clothing (a white dress, say),
    so when the background is replaced the backdrop bleeds through the cloth ("mixing"). We
    fill interior holes and force the sure-foreground core to alpha 1.0, while leaving the
    outer band as rembg's soft alpha so hair/edges still blend naturally.
    """
    h, w = alpha.shape
    solid = (alpha > 0.5).astype(np.uint8) * 255
    # fill holes: flood the background in from every corner, whatever stays 0 is an interior hole
    ff = solid.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[seed[1], seed[0]] == 0:
            cv2.floodFill(ff, ffmask, seed, 255)
    filled = cv2.bitwise_or(solid, cv2.bitwise_not(ff))
    # sure-foreground = filled core eroded inward; force it opaque
    k = max(3, int(max(h, w) * 0.01)) | 1
    interior = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return np.clip(np.maximum(alpha, interior.astype(np.float32) / 255.0), 0, 1)


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
        alpha = _solidify_alpha(alpha)               # stop light clothing going transparent
        return np.clip(_feather(alpha, max(h, w) * 0.0015), 0, 1)
    except Exception as exc:
        log.warning("rembg segmentation failed: %s", exc)
        return None


# --------------------------------------------------------------------- operations

def _soften_skin(img: np.ndarray, skin: np.ndarray, strength: float,
                 face_w: float) -> np.ndarray:
    """Frequency separation: even skin tone AND attenuate fine texture proportional to strength.

    detail (pores, fine texture, noise) = img - base; the base is evened with an edge-preserving
    bilateral filter. The old version added detail back at 100%, so higher --skin was invisible.
    Now the finest detail is scaled by `keep` (lower as strength rises) so the skin visibly
    softens, while a floor keeps enough pore texture that it never looks plastic. Coarser detail
    (above the pore scale) is preserved, so edges and features stay sharp.
    """
    if strength <= 0.01 or skin.max() < 0.01:
        return img
    s = min(max(strength, 0.0), 1.0)

    sigma = max(2.0, face_w * 0.016)
    base = cv2.GaussianBlur(img, (0, 0), sigma)
    detail = img - base

    # split detail into coarse (kept — features/edges) and fine (pores/noise — attenuated)
    fine_sigma = max(1.0, face_w * 0.004)
    detail_coarse = cv2.GaussianBlur(detail, (0, 0), fine_sigma)   # smooth part of the detail
    detail_fine = detail - detail_coarse                          # the finest texture / pores

    # bilateral on a downscaled base (base has no fine detail, nothing is lost)
    h, w = img.shape[:2]
    scale = min(1.0, 1400 / max(h, w))
    small = cv2.resize(base, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 9, 0.08, 8)
    smooth_base = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    keep = 1.0 - 0.75 * s                       # s=0.3->0.78, 0.5->0.63, 0.9->0.33 fine texture kept
    softened = smooth_base + detail_coarse + detail_fine * keep
    m = skin[..., None]                         # apply fully inside the (feathered) skin mask
    return img * (1 - m) + softened * m


def _tame_skin_highlights(img: np.ndarray, skin: np.ndarray, face_w: float,
                          strength: float) -> np.ndarray:
    """Recover overexposed / shiny skin: pull the brightest skin areas down toward the
    surrounding skin tone so blown patches regain shape and color instead of reading as
    flat white. Only bright skin is affected; normal skin and non-skin are untouched."""
    if strength <= 0.01 or skin.max() < 0.01:
        return img
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    # local skin tone/brightness (broad blur), the target the hot spots get pulled toward
    base = cv2.GaussianBlur(img, (0, 0), max(3.0, face_w * 0.05))
    over = np.clip((lum - 0.78) / 0.22, 0, 1) ** 1.5      # 0 below ~0.78, ramps to blown
    w = (skin * over * min(max(strength, 0.0), 1.0))[..., None]
    target = np.minimum(img, base)                        # only ever darken toward local tone
    return np.clip(img * (1 - w) + target * w, 0, 1)


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


# Healthy skin target per type, as a HUE ANGLE in CIELAB (degrees of atan2(b*, a*)) plus a target
# chroma and the max hue rotation we'll allow. These are NATURAL skin-hue references (~55-58°, the
# healthy skin line) — not pushed toward orange. The correction nudges skin toward its type's
# natural centre and pulls colour casts (green/sallow) back onto the line, keeping the tone the
# camera captured rather than restyling it. Chroma targets restore a healthy saturation without
# oversaturating. (Warmth can bias these toward orange via --skin-warmth, but it defaults to 0.)
_SKIN_TONE_TARGETS = {
    #  type   : (target hue°, target chroma, max rotation°)
    "fair":    (55.0, 16.0, 12.0),
    "light":   (56.0, 18.0, 12.0),
    "medium":  (57.0, 21.0, 14.0),
    "olive":   (58.0, 21.0, 16.0),   # olive: correct the green undertone, stay natural (not orange)
    "tan":     (58.0, 23.0, 14.0),
    "brown":   (57.0, 25.0, 14.0),
    "deep":    (56.0, 24.0, 14.0),
}
_SKIN_TYPE_ALIASES = {
    "very_light": "fair", "pale": "fair", "type1": "fair", "type2": "light",
    "type3": "medium", "type4": "olive", "type5": "brown", "type6": "deep",
    "dark": "deep", "ebony": "deep", "black": "deep", "tanned": "tan",
}


def _classify_skin_type(l_med: float, b_med: float) -> str:
    """Classify skin by ITA° (Individual Typology Angle) — the dermatology-standard metric,
    computed from lightness and yellowness: ITA = atan2(L*-50, b*). Higher = lighter/cooler."""
    ita = np.degrees(np.arctan2(l_med - 50.0, max(b_med, 1e-3)))
    if ita > 45:
        return "fair"
    if ita > 30:
        return "light"
    if ita > 15:
        return "medium"
    if ita > 0:
        return "tan"
    if ita > -30:
        return "brown"
    return "deep"


def _skin_tone_correction(img: np.ndarray, skin: np.ndarray, strength: float,
                          skin_type: str = "auto", warmth: float = 0.0,
                          saturation: float = 1.0, red: float = 0.0) -> np.ndarray:
    """Even out blotchy skin and rotate it toward a warm, orange-leaning tone for its skin type.

    Works on the skin's hue: the whole chroma cloud is rotated toward the type's target hue
    (~48-50°, the orange side of skin) so yellow/sallow skin gains orange, and its chroma is
    scaled toward a healthy level. `warmth` (-1..1) biases the target further toward orange
    (+) or lets it stay yellower (-). Luminance and per-pixel variation are preserved; the
    rotation is capped per type so skin never turns ruddy/sunburnt.
    """
    if strength <= 0.01 or skin.max() < 0.01:
        return img
    h, w = img.shape[:2]
    s = min(max(strength, 0.0), 1.0)
    warmth = float(np.clip(warmth, -1.0, 1.0))
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    sel = skin > 0.5
    if not sel.any():
        return img

    # 1. even out red/green blotches (smooth chroma only; texture lives in luminance)
    ev = skin * s * 0.6
    a = a * (1 - ev) + cv2.GaussianBlur(a, (0, 0), max(h, w) * 0.005) * ev
    b = b * (1 - ev) + cv2.GaussianBlur(b, (0, 0), max(h, w) * 0.005) * ev

    # 2. classify (or take the given type) from the evened median chroma/lightness
    st = (skin_type or "auto").strip().lower()
    st = _SKIN_TYPE_ALIASES.get(st, st)
    l_med, a_med, b_med = float(np.median(L[sel])), float(np.median(a[sel])), float(np.median(b[sel]))
    if st == "auto" or st not in _SKIN_TONE_TARGETS:
        st = _classify_skin_type(l_med, b_med)
    t_hue, t_chroma, max_rot = _SKIN_TONE_TARGETS[st]
    t_hue -= warmth * 12.0                          # +warmth -> lower hue = more orange

    # 3. rotate each skin pixel toward the target hue (PER-PIXEL, not one global angle) and scale
    #    chroma to target. Pixels on the yellow/green side of the target are pulled hard toward it
    #    — this erases localized green/sallow patches (common in shadowed skin) that a single
    #    uniform rotation leaves behind — while pixels on the red side move only gently, so natural
    #    blush and lip warmth survive.
    h0 = float(np.degrees(np.arctan2(b_med, a_med)))
    c0 = float(np.hypot(a_med, b_med))
    hue_px = np.degrees(np.arctan2(b, a))
    # Pull the yellow/green side toward the skin hue (removes casts), but leave the RED side
    # essentially untouched — the naturally red lips and cheek blush must keep their color, not be
    # rotated toward yellow-skin. (This is why cheeks/lips were losing their red before.)
    reach = np.where(hue_px > t_hue, min(0.7, 0.35 + 0.4 * s), 0.04)
    d_deg = np.clip((t_hue - hue_px) * reach, -50.0, max_rot)
    rad = np.radians(d_deg)
    cos, sin = np.cos(rad), np.sin(rad)
    cs = 1.0 + (t_chroma / max(c0, 1e-3) - 1.0) * (0.5 * s)
    cs = min(1.6, max(0.7, cs)) + max(0.0, warmth) * 0.1
    cs = min(1.9, cs * max(0.5, saturation))    # richness: deepen skin color so it isn't flat

    a_rot = (a * cos - b * sin) * cs                # rotate toward orange, then scale chroma
    b_rot = (a * sin + b * cos) * cs
    if red > 0.001:
        # add a healthy red flush (raise a* in the midtones, where cheeks live) so skin has the
        # warm red the camera's colour science shows, without reddening shadows or hot highlights
        wL = np.exp(-((L - 50.0) / 35.0) ** 2)
        a_rot = a_rot + red * 7.0 * wL
    lab[..., 1] = a * (1 - skin) + a_rot * skin
    lab[..., 2] = b * (1 - skin) + b_rot * skin
    hue_out = float(np.degrees(np.arctan2(np.median(b_rot[sel]), np.median(a_rot[sel]))))
    log.info("skin tone: type=%s hue %.0f°->%.0f° (per-pixel, chroma x%.2f, warmth %+.1f)",
             st, h0, hue_out, cs, warmth)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _brighten_skin(img: np.ndarray, skin: np.ndarray, amount: float) -> np.ndarray:
    """Lift skin luminance the way the Lightroom Color-Mixer luminance sliders do for portraits:
    a base brightening across all skin (like Orange Luminance +12) plus extra on the redder skin —
    lips and cheeks — (like Red Luminance +31), for brighter, more vital skin. Only luminance is
    touched (colour and texture are left to the tone/soften steps); highlights are protected."""
    if amount <= 0.01 or skin.max() < 0.01:
        return img
    s = min(max(amount, 0.0), 1.0)
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    L, a = lab[..., 0], lab[..., 1]
    red_w = np.clip((a - 8.0) / 12.0, 0, 1)             # 0 on normal skin, 1 on lips/red cheeks
    headroom = np.clip((100.0 - L) / 100.0, 0, 1)       # don't push near-white skin to pure white
    # modest extra on reds only (heavy lightening washes out lip/cheek red — that's handled by
    # the lip enhancement instead), mostly an even skin brightening.
    lift = skin * s * (7.0 + 3.0 * red_w) * headroom    # ~+3.5 base, up to ~+5 on reds at s=0.5
    lab[..., 0] = np.clip(L + lift, 0, 100)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _hair_mask(img: np.ndarray, faces: list[Face], subject: np.ndarray | None,
               skin: np.ndarray) -> np.ndarray | None:
    """Approximate the hair region: subject pixels around/above the face that aren't skin."""
    if not faces or subject is None:
        return None
    h, w = img.shape[:2]
    region = np.zeros((h, w), np.float32)
    for f in faces:
        cx, cy = f.x + f.w / 2.0, f.y + f.h * 0.15
        cv2.ellipse(region, (int(cx), int(cy)),
                    (int(f.w * 1.15), int(f.h * 1.35)), 0, 0, 360, 1.0, -1)
    face_w = max(f.w for f in faces)
    hair = _feather(region * subject * (1.0 - np.clip(skin, 0, 1)), face_w * 0.02)
    return np.clip(hair, 0, 1) if hair.max() > 0.02 else None


def _enhance_hair(img: np.ndarray, hair: np.ndarray | None, subject: np.ndarray | None,
                  texture: float, shimmer: float, defrizz: float, face_w: float) -> np.ndarray:
    """Make hair strands pop and (optionally) add shimmer and remove frizz.

    - texture: two-scale local contrast — fine detail (Texture, strand pop) + a mid-scale
      boost (Clarity), applied only inside the hair mask.
    - shimmer: lifts highlights and deepens shadows within the hair for a natural sheen.
    - defrizz: trims flyaway strands by painting the background over stray hairs that stick
      out past a smoothed hair silhouette (a straighter, cleaner look).
    """
    if hair is None or hair.max() < 0.02:
        return img
    out = img
    m = hair[..., None]

    if texture > 0.01:
        s = min(max(texture, 0.0), 1.0)
        detail_tex = img - cv2.GaussianBlur(img, (0, 0), max(1.0, face_w * 0.004))   # Texture
        detail_cla = img - cv2.GaussianBlur(img, (0, 0), max(3.0, face_w * 0.02))    # Clarity
        boosted = np.clip(img + detail_tex * (0.9 * s) + detail_cla * (0.5 * s), 0, 1)
        out = out * (1 - m) + boosted * m

    if shimmer > 0.01:
        s = min(max(shimmer, 0.0), 1.0)
        lab = cv2.cvtColor(np.clip(out, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
        L = lab[..., 0] / 100.0
        hi = np.clip((L - 0.6) / 0.4, 0, 1)     # highlights up (+5..10 feel)
        lo = np.clip((0.4 - L) / 0.4, 0, 1)     # shadows down (-5..10 feel)
        lab[..., 0] = np.clip((L + hair * s * (hi - lo) * 0.08) * 100.0, 0, 100)
        out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)

    if defrizz > 0.01 and subject is not None:
        s = min(max(defrizz, 0.0), 1.0)
        k = max(3, int(face_w * 0.02)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        smooth = cv2.GaussianBlur(cv2.morphologyEx(subject, cv2.MORPH_OPEN, kernel),
                                  (0, 0), face_w * 0.01)
        frizz = np.clip(subject - smooth, 0, 1)                      # bits outside silhouette
        gate = cv2.dilate(hair, np.ones((k, k), np.uint8))          # only near the hair
        bg = _normalized_blur(out, subject, face_w * 0.03)          # background colors
        fm = (frizz * gate * s)[..., None]
        out = out * (1 - fm) + bg * fm

    return out


def _enhance_lips(img: np.ndarray, faces: list[Face], strength: float) -> np.ndarray:
    """Give the lips a natural, fuller look: restore/deepen their red, add a little richness and
    definition. Detected inside the mouth region by lip colour (reddish, mid-tone), so teeth,
    braces and the dark mouth interior are left alone. This is what portrait retouchers do to the
    lips after skin work; it also puts back the lip red that skin correction can flatten."""
    if strength <= 0.01 or not faces:
        return img
    h, w = img.shape[:2]
    face_w = max(f.w for f in faces)
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    roi = np.zeros((h, w), np.uint8)      # tight mouth region (rotated to the mouth axis)
    ring = np.zeros((h, w), np.uint8)     # skin just around the mouth, to learn the skin colour
    for f in faces:
        mx = (f.mouth_left[0] + f.mouth_right[0]) / 2
        my = (f.mouth_left[1] + f.mouth_right[1]) / 2
        mw = np.hypot(f.mouth_right[0] - f.mouth_left[0], f.mouth_right[1] - f.mouth_left[1])
        if mw < 4:
            continue
        ang = np.degrees(np.arctan2(f.mouth_right[1] - f.mouth_left[1],
                                    f.mouth_right[0] - f.mouth_left[0]))
        cv2.ellipse(roi, (int(mx), int(my)), (int(mw * 0.62), int(mw * 0.30)), ang, 0, 360, 1, -1)
        cv2.ellipse(ring, (int(mx), int(my)), (int(mw * 1.5), int(mw * 1.0)), ang, 0, 360, 1, -1)
    roi_b = roi > 0
    if int(roi_b.sum()) < 30:
        return img
    # learn the surrounding skin's redness, then keep only pixels distinctly REDDER than skin —
    # that isolates the actual lip tone and rejects perioral skin, teeth and braces (near-neutral)
    ring_b = (ring > 0) & (~(cv2.dilate(roi, np.ones((9, 9), np.uint8)) > 0))
    peri_a = float(np.median(a[ring_b])) if int(ring_b.sum()) > 50 else 9.0
    lip = (roi_b & (a > peri_a + 5.0) & (L > 24.0) & (L < 74.0)).astype(np.float32)
    lip = _feather(lip, face_w * 0.006)
    if lip.max() < 0.02:
        return img

    s = min(max(strength, 0.0), 1.0)
    m = lip                                            # 2-D blend weight (per Lab channel below)
    csat = 1.0 + 0.55 * s                              # richer, more saturated lip colour
    a_e = a * csat + 3.5 * s                           # a touch more red
    b_e = b * csat
    L_e = L - 2.5 * s                                  # slightly deeper for a fuller look
    lab[..., 0] = L * (1 - m) + np.clip(L_e, 0, 100) * m
    lab[..., 1] = a * (1 - m) + a_e * m
    lab[..., 2] = b * (1 - m) + b_e * m
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


# Fabric whose color should NOT be saturated — boosting a near-neutral garment only amplifies
# noise and any color cast. Tonal moves (rich blacks, clean whites) still apply to these.
_NEUTRAL_CLOTH = {"", "none", "black", "white", "gray", "grey", "silver", "charcoal", "ivory"}


def _clothing_color_grade(img: np.ndarray, subject: np.ndarray | None, skin: np.ndarray,
                          hair: np.ndarray | None, color: str, color_pop: float,
                          luminance: float, shadows: float, blacks: float, whites: float,
                          mult: float) -> np.ndarray:
    """Color-aware tonal grade of the clothing region (subject minus skin and hair).

    Releases the fabric color (vibrance-weighted chroma boost) and applies the usual develop
    moves — luminance, shadows, blacks, whites — but only inside the cloth. The AI reports the
    dominant `color`; for neutral fabric (black/white/gray) the color pop is suppressed while
    the tonal moves still apply (deep rich blacks, clean bright whites). Luminance texture is
    left untouched; everything blends through the feathered cloth mask so nothing else shifts.
    """
    color_pop = max(0.0, color_pop) * mult
    luminance = float(np.clip(luminance, -1, 1)) * mult
    shadows = float(np.clip(shadows, -1, 1)) * mult
    blacks = float(np.clip(blacks, -1, 1)) * mult
    whites = float(np.clip(whites, -1, 1)) * mult
    if max(abs(color_pop), abs(luminance), abs(shadows), abs(blacks), abs(whites)) < 0.01:
        return img
    if subject is None or subject.max() < 0.05:
        log.warning("Clothing grade requested but no subject matte — skipping")
        return img

    cloth = np.clip(subject, 0, 1) * (1.0 - np.clip(skin, 0, 1))
    if hair is not None:
        cloth = cloth * (1.0 - np.clip(hair, 0, 1))
    if cloth.max() < 0.02:
        return img

    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    Lg = lab[..., 0] / 100.0
    a, b = lab[..., 1], lab[..., 2]

    # tonal moves on the L channel (bounded, gentle) — the standard develop zones
    if abs(luminance) > 0.01:
        Lg = Lg * (1.0 + 0.45 * luminance)                 # overall brighten / darken
    if abs(shadows) > 0.01:
        lo = Lg < 0.5
        Lg[lo] = Lg[lo] + shadows * (0.5 - Lg[lo])         # lift(+) / deepen(-) the lower half
    if abs(blacks) > 0.01:
        w = np.clip((0.30 - Lg) / 0.30, 0, 1)              # weight toward the darkest tones
        Lg = Lg + blacks * 0.18 * w                        # black point: -0.2 = rich deep blacks
    if abs(whites) > 0.01:
        w = np.clip((Lg - 0.70) / 0.30, 0, 1)              # weight toward the brightest tones
        Lg = Lg + whites * 0.18 * w                        # white point: - recovers, + brightens
    Lg = np.clip(Lg, 0, 1)

    # color pop: vibrance-weighted chroma boost, suppressed for neutral fabric
    if color_pop > 0.01 and (color or "").strip().lower() not in _NEUTRAL_CLOTH:
        chroma = np.hypot(a, b)
        vib = 1.0 - np.clip(chroma / 45.0, 0, 1)           # less boost where already saturated
        scale = 1.0 + color_pop * (0.35 + 0.55 * vib)      # up to ~1.5x on a flat, dull color
        a, b = a * scale, b * scale

    lab[..., 0], lab[..., 1], lab[..., 2] = Lg * 100.0, a, b
    graded = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    m = cloth[..., None]
    return np.clip(img * (1 - m) + graded * m, 0, 1)


def _expose_subject(img: np.ndarray, subject: np.ndarray | None, ev: float) -> np.ndarray:
    """Raise (or lower) the exposure of only the subject, leaving the background untouched.

    `ev` is in stops (like the develop exposure): +1.0 doubles the subject's brightness,
    -1.0 halves it. The gain is blended with a highlight rolloff so already-bright areas
    of the subject don't harshly clip, and feathered by the subject mask (no edge halo).
    """
    if abs(ev) < 0.01:
        return img
    if subject is None or subject.max() < 0.05:
        log.warning("Subject exposure requested but no subject mask — skipping")
        return img
    gain = 2.0 ** ev
    boosted = img * gain
    if gain > 1.0:
        # soft-clip: ease the top end so highlights roll off instead of blowing out
        boosted = np.where(boosted > 0.8, 0.8 + (boosted - 0.8) / gain, boosted)
    m = subject[..., None]
    return np.clip(img * (1 - m) + boosted * m, 0, 1)


def _expose_background(img: np.ndarray, subject: np.ndarray | None, ev: float) -> np.ndarray:
    """Raise (or lower) the exposure of only the background, leaving the subject untouched.

    Same soft-clipped gain as `_expose_subject`, but applied to (1 - subject mask)."""
    if abs(ev) < 0.01:
        return img
    if subject is None or subject.max() < 0.05 or subject.mean() > 0.95:
        log.warning("Background exposure requested but no usable subject mask — skipping")
        return img
    gain = 2.0 ** ev
    boosted = img * gain
    if gain > 1.0:
        boosted = np.where(boosted > 0.8, 0.8 + (boosted - 0.8) / gain, boosted)
    m = (1.0 - subject)[..., None]
    return np.clip(img * (1 - m) + boosted * m, 0, 1)


def _expose_masked(img: np.ndarray, mask: np.ndarray, ev: float) -> np.ndarray:
    """Apply a soft-clipped exposure gain of `ev` stops within `mask` (feathered blend)."""
    if abs(ev) < 0.01 or mask.max() < 0.01:
        return img
    gain = 2.0 ** ev
    boosted = img * gain
    if gain > 1.0:                      # ease the top end so highlights roll off, not clip
        boosted = np.where(boosted > 0.8, 0.8 + (boosted - 0.8) / gain, boosted)
    m = np.clip(mask, 0, 1)[..., None]
    return np.clip(img * (1 - m) + boosted * m, 0, 1)


def _meter_region_ev(img: np.ndarray, mask: np.ndarray, target: float, max_ev: float) -> float:
    """Exposure correction (stops) that moves a region's median luminance toward `target`.

    Metering on the median (not mean) makes it robust to specular highlights and shadows.
    The result is clamped to +/- max_ev so a mis-metered region can never overcook."""
    sel = mask > 0.5
    if not sel.any():
        return 0.0
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    cur = float(np.median(lum[sel]))
    if cur <= 1e-4:
        return 0.0
    ev = float(np.log2(max(target, 1e-3) / cur))
    return float(np.clip(ev, -max_ev, max_ev))


def _auto_levels(img: np.ndarray, skin: np.ndarray, subject: np.ndarray | None,
                 faces: list[Face], skin_target: float, subject_target: float,
                 bg_target: float, max_ev: float) -> np.ndarray:
    """Per-layer auto exposure: meter face-skin, the rest of the subject, and the background
    each toward their own midtone target and correct through disjoint masks.

    The three layers are disjoint (skin / subject-minus-skin / background) so a per-layer gain
    only touches that layer — no double-correction. Face-skin is metered first and, if it still
    has blown pixels afterward, `_tame_skin_highlights` recovers them, so a single flag fixes
    the common 'overexposed skin' failure end-to-end.
    """
    face_w = max((f.w for f in faces), default=0.0)

    # 1. face/neck skin -> target midtone, then recover any remaining blown skin
    if skin_target > 0 and faces and skin.max() > 0.05:
        ev = _meter_region_ev(img, skin, skin_target, max_ev)
        if abs(ev) > 0.01:
            img = _expose_masked(img, skin, ev)
            log.info("auto-levels: skin %+.2f EV toward %.2f", ev, skin_target)
        lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        sel = skin > 0.5
        blown = float((lum[sel] > 0.92).mean()) if sel.any() else 0.0
        if blown > 0.01:
            strength = min(1.0, blown * 8.0)
            img = _tame_skin_highlights(img, skin, face_w, strength)
            log.info("auto-levels: %.1f%% skin still blown -> tame highlights %.2f",
                     100 * blown, strength)

    # 2. rest of the subject (body / clothing / hair), excluding face-skin -> target
    if subject_target > 0 and subject is not None and subject.max() > 0.05:
        body = np.clip(subject - skin, 0, 1)
        if body.max() > 0.05:
            ev = _meter_region_ev(img, body, subject_target, max_ev)
            if abs(ev) > 0.01:
                img = _expose_masked(img, body, ev)
                log.info("auto-levels: subject %+.2f EV toward %.2f", ev, subject_target)

    # 3. background -> target (only meaningful when a subject actually splits the frame)
    if bg_target > 0 and subject is not None and 0.05 < float(subject.mean()) < 0.95:
        bg = np.clip(1.0 - subject, 0, 1)
        ev = _meter_region_ev(img, bg, bg_target, max_ev)
        if abs(ev) > 0.01:
            img = _expose_masked(img, bg, ev)
            log.info("auto-levels: background %+.2f EV toward %.2f", ev, bg_target)

    return img


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


# named studio backdrop colors (base RGB in 0-1; a soft gradient is applied on top)
_STUDIO_COLORS = {
    "gray": (0.56, 0.56, 0.57), "grey": (0.56, 0.56, 0.57),
    "lightgray": (0.74, 0.74, 0.75), "lightgrey": (0.74, 0.74, 0.75),
    "white": (0.93, 0.93, 0.94), "charcoal": (0.24, 0.24, 0.26), "black": (0.10, 0.10, 0.11),
    "blue": (0.19, 0.27, 0.44), "navy": (0.12, 0.16, 0.30), "teal": (0.16, 0.34, 0.38),
    "green": (0.24, 0.36, 0.28), "red": (0.44, 0.15, 0.17), "maroon": (0.30, 0.11, 0.14),
    "pink": (0.82, 0.66, 0.71), "purple": (0.32, 0.24, 0.42), "beige": (0.80, 0.74, 0.63),
    "brown": (0.36, 0.28, 0.22),
}


def _resolve_studio_color(color: str) -> tuple[float, float, float]:
    """Named color or #hex -> RGB tuple in 0-1. Falls back to gray."""
    c = (color or "gray").strip().lower()
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            except ValueError:
                pass
    if c not in _STUDIO_COLORS:
        log.warning("Unknown studio color %r — using gray", color)
    return _STUDIO_COLORS.get(c, _STUDIO_COLORS["gray"])


def _studio_background(img: np.ndarray, alpha: np.ndarray, color: str = "gray") -> np.ndarray:
    """Replace the background with a smooth studio backdrop of the given color."""
    h, w = img.shape[:2]
    base = np.array(_resolve_studio_color(color), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 - 0.26 * (yy / h)                        # brighter top, darker bottom
    r = np.hypot((xx - w / 2) / w, (yy - h * 0.4) / h)   # gentle vignette
    shade = np.clip(shade - 0.12 * r, 0.0, 1.15)
    bg = np.clip(base[None, None, :] * shade[..., None], 0, 1)
    a = alpha[..., None]
    return np.clip(img * a + bg * (1 - a), 0, 1)


def _comfy_inpaint(img: np.ndarray, regen: np.ndarray, prompt: str) -> np.ndarray:
    """Regenerate the regions where `regen` (float 0..1, 1 = regenerate) is set, via
    ComfyUI inpainting, and return the full-size result image. The caller decides how
    to composite it (background replace keeps the subject; erase keeps everything
    outside the drawn mask)."""
    import httpx

    workflow = json.loads((_WORKFLOW_DIR / "inpaint.json").read_text())["nodes"]

    h, w = img.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    sw, sh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(regen, (sw, sh), interpolation=cv2.INTER_LINEAR)

    tmp = CONFIG.output / f".comfy_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(exist_ok=True)
    img_path = tmp / "input.png"
    mask_path = tmp / "mask.png"
    Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8)).save(img_path)
    # mask: white = regenerate, slightly grown so the seam lands outside the kept pixels
    mask_u8 = cv2.dilate(np.clip(small_mask * 255, 0, 255).astype(np.uint8),
                         np.ones((5, 5), np.uint8))
    Image.fromarray(np.stack([mask_u8] * 3, axis=-1)).save(mask_path)

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
        log.info("Generative inpaint submitted (prompt_id=%s)", prompt_id)

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
    return cv2.resize(generated, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _replace_background_comfy(img: np.ndarray, alpha: np.ndarray, prompt: str) -> np.ndarray:
    """Generative background via ComfyUI inpainting. Only the background region is
    regenerated; the full-resolution subject is composited back over the result."""
    generated = _comfy_inpaint(img, 1.0 - alpha, prompt)
    a = alpha[..., None]
    return np.clip(img * a + generated * (1 - a), 0, 1)


# ------------------------------------------------------------------ object erase

def _load_erase_mask(name: str, shape: tuple) -> np.ndarray | None:
    """Load an operator-drawn erase mask (white = remove) and fit it to the image."""
    if not name:
        return None
    path = Path(name)
    if not path.is_absolute():
        path = CONFIG.root / "masks" / path
    if not path.exists():
        log.warning("Erase mask %s not found — skipping erase", path)
        return None
    m = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m if m.max() > 0.05 else None


def _erase_regions(img: np.ndarray, mask: np.ndarray, method: str, prompt: str) -> np.ndarray:
    """Remove the masked parts of the photo and fill the hole from its surroundings.

    method 'content-aware': deterministic OpenCV inpainting (Photoshop's classic
    Content-Aware Fill idea) run per masked region at full resolution — fast, local,
    great for wrinkles, cables, lint, small distractions.
    method 'generative': ComfyUI diffusion inpaint (Generative Fill) — regenerates the
    masked area from the surrounding scene; better for large objects, but needs ComfyUI.
    """
    hard = (mask > 0.5).astype(np.uint8)
    if not hard.any():
        return img
    if method == "generative":
        generated = _comfy_inpaint(img, mask, prompt or
                                   "empty scene, seamless continuation of the surrounding "
                                   "background, nothing there")
        m = _feather(np.clip(mask, 0, 1), 3.0)[..., None]
        return np.clip(img * (1 - m) + generated * m, 0, 1)

    # content-aware: inpaint each region's bounding box (with margin) so a huge photo
    # doesn't pay full-frame cost; big boxes are inpainted downscaled, then upsampled.
    out = img.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    h, w = hard.shape
    for i in range(1, n):
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                       stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        pad = max(24, int(0.4 * max(bw, bh)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        roi = out[y0:y1, x0:x1]
        roi_mask = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        # grow a touch so no halo of the removed object survives at the edge
        roi_mask = cv2.dilate(roi_mask, np.ones((5, 5), np.uint8))
        scale = min(1.0, 1600 / max(roi.shape[0], roi.shape[1]))
        if scale < 1.0:
            sm = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            sm_mask = cv2.resize(roi_mask, (sm.shape[1], sm.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            filled = cv2.inpaint((np.clip(sm, 0, 1) * 255).astype(np.uint8),
                                 sm_mask, 5, cv2.INPAINT_TELEA)
            filled = cv2.resize(filled, (roi.shape[1], roi.shape[0]),
                                interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
        else:
            filled = cv2.inpaint((np.clip(roi, 0, 1) * 255).astype(np.uint8),
                                 roi_mask, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        m = _feather(roi_mask.astype(np.float32), 2.0)[..., None]
        out[y0:y1, x0:x1] = roi * (1 - m) + filled * m
    return out


def _auto_background(img: np.ndarray, alpha: np.ndarray) -> tuple[str, str]:
    """Inspect the background and choose an action + color deterministically:

    - a neutral studio backdrop that fills the frame but has fold wrinkles -> ('smooth', color)
    - a neutral backdrop that only partly fills the frame (other zones visible, e.g. floor,
      stands, wall) -> ('studio', color) so those zones are replaced with a clean backdrop
    - anything else (a real scene, or a clean seamless backdrop) -> ('keep', color)

    color is the backdrop's neutral tone: 'white', 'gray' or 'black' from its median lightness.
    """
    h, w = img.shape[:2]
    bg = alpha < 0.5
    n = int(bg.sum())
    if n < 0.02 * h * w:                       # subject fills the frame — no background to speak of
        return "keep", "gray"

    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    neutral = bg & (chroma < CONFIG.auto_bg_neutral_chroma)
    if int(neutral.sum()) < 0.30 * n:          # background isn't a neutral backdrop -> real scene
        log.info("auto-bg: background not neutral (%.0f%% neutral) -> keep", 100 * neutral.sum() / n)
        return "keep", "gray"

    l_dom = float(np.median(L[neutral]))
    color = "white" if l_dom > 78 else ("black" if l_dom < 28 else "gray")
    backdrop = neutral & (np.abs(L - l_dom) < 18.0)     # the dominant backdrop tone
    backdrop_frac = backdrop.sum() / max(1, n)

    # wrinkle score: mid-frequency texture in the backdrop (folds/creases), lighting gradient removed
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = g - cv2.GaussianBlur(g, (0, 0), max(h, w) * 0.02)
    wrinkle = float(detail[backdrop].std()) if int(backdrop.sum()) > 100 else 0.0

    if backdrop_frac > 0.90:
        action = "smooth" if wrinkle > CONFIG.auto_bg_wrinkle_thresh else "keep"
    elif backdrop_frac > 0.50:
        action = "studio"                       # backdrop + other zones -> unify with a studio color
    else:
        action = "keep"
    log.info("auto-bg: backdrop_frac=%.2f L=%.0f (%s) wrinkle=%.1f -> %s",
             backdrop_frac, l_dom, color, wrinkle, action)
    return action, color


def _apply_background(img: np.ndarray, action: str, prompt: str,
                      alpha: np.ndarray | None, color: str = "gray") -> np.ndarray:
    if action == "keep":
        return img
    if alpha is None or alpha.max() < 0.05 or alpha.mean() > 0.95:
        log.warning("No usable subject mask — leaving background unchanged")
        return img
    if action == "auto":
        action, color = _auto_background(img, alpha)
        if action == "keep":
            return img
    if action == "blur":
        return _blur_background(img, alpha)
    if action == "smooth":
        return _smooth_background(img, alpha)
    if action == "studio":
        return _studio_background(img, alpha, color)
    if action == "replace":
        try:
            return _replace_background_comfy(img, alpha, prompt or "clean neutral studio backdrop")
        except RetouchError as exc:
            log.warning("Generative replace failed (%s) — using studio backdrop instead", exc)
            return _studio_background(img, alpha, color)
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

    wants_hair = rp.get("is_portrait", False) and any((
        rp.get("hair_texture", 0) > 0.01,
        rp.get("hair_shimmer", 0) > 0.01,
        rp.get("hair_defrizz", 0) > 0.01,
    ))
    wants_face_work = rp.get("is_portrait", False) and (wants_hair or any((
        rp.get("skin_smoothing", 0) > 0.01,
        rp.get("remove_blemishes", False),
        rp.get("brighten_eyes", 0) > 0.01,
        rp.get("whiten_teeth", 0) > 0.01,
        rp.get("iris_enhance", 0) > 0.01,
        rp.get("reduce_dark_circles", 0) > 0.01,
        rp.get("reduce_dewlap", 0) > 0.01,
        rp.get("skin_tone_correction", 0) > 0.01,
        rp.get("tame_highlights", 0) > 0.01,
        rp.get("skin_luminance", 0) > 0.01,
        rp.get("lip_enhance", 0) > 0.01,
    )))
    clothing = rp.get("clothing", {}) or {}
    wants_cloth_grade = any(abs(float(clothing.get(k, 0) or 0)) > 0.01
                            for k in ("color_pop", "luminance", "shadows", "blacks", "whites"))
    wants_cloth = rp.get("clothing_contrast", 0) > 0.01 or wants_cloth_grade
    wants_bg = bg_action != "keep"
    erase = rp.get("erase", {}) or {}
    wants_erase = bool(erase.get("mask"))
    subject_ev = float(rp.get("subject_exposure", 0.0))
    background_ev = float(rp.get("background_exposure", 0.0))
    wants_subject_exp = abs(subject_ev) > 0.01
    wants_bg_exp = abs(background_ev) > 0.01

    # Auto-levels: opt-in per-layer exposure metering. Targets fall back to CONFIG defaults;
    # 0 disables a layer. Only face-skin metering needs faces; subject/background need a matte.
    auto_levels = bool(rp.get("auto_levels", False))
    skin_target = float(rp.get("auto_skin_target", CONFIG.auto_levels_skin_target)) if auto_levels else 0.0
    subject_target = float(rp.get("auto_subject_target", CONFIG.auto_levels_subject_target)) if auto_levels else 0.0
    bg_target = float(rp.get("auto_background_target", CONFIG.auto_levels_background_target)) if auto_levels else 0.0
    wants_auto = auto_levels and (skin_target > 0 or subject_target > 0 or bg_target > 0)
    wants_auto_skin = wants_auto and skin_target > 0
    wants_auto_subject = wants_auto and (subject_target > 0 or bg_target > 0)

    if CONFIG.skip_retouch or not (wants_face_work or wants_cloth or wants_bg
                                   or wants_subject_exp or wants_bg_exp or wants_auto
                                   or wants_erase):
        log.info("Skipping retouch (nothing requested, skip=%s)", CONFIG.skip_retouch)
        return tiff_path

    mult = _INTENSITY.get(rp.get("intensity", "natural"), 1.0)
    img = _load_image(tiff_path)

    # erase first: every later op (subject matte, masks, exposure) must see the
    # cleaned frame, not the removed object
    if wants_erase:
        erase_mask = _load_erase_mask(erase.get("mask", ""), img.shape[:2])
        if erase_mask is not None:
            img = _erase_regions(img, erase_mask, erase.get("method", "content-aware"),
                                 erase.get("prompt", ""))
            log.info("Erased %.1f%% of the frame (%s)", 100 * float((erase_mask > 0.5).mean()),
                     erase.get("method", "content-aware"))

    need_faces = wants_face_work or wants_auto_skin
    faces = _detect_faces(img) if need_faces else []
    if wants_face_work and not faces:
        log.warning("Analysis says portrait but no face detected — skipping face retouch")

    # Subject matte first: face-work needs it too, so body skin (arms/shoulders) is tone-matched
    # to the face. In practice portraits already trigger it (hair enhancement is on by default).
    alpha = _subject_alpha(img) if (wants_bg or wants_cloth or wants_hair
                                    or wants_subject_exp or wants_bg_exp
                                    or wants_auto_subject or wants_face_work) else None
    # skin_tone = ALL skin incl. eyes/lips (tone correction reaches the whole face evenly);
    # skin = the same with eyes/lips removed, for ops that must not blur/clone those features.
    skin_tone = _skin_mask(img, faces, alpha) if faces else np.zeros(img.shape[:2], np.float32)
    skin = (np.clip(skin_tone * (1.0 - _feature_exclusion(img.shape, faces, max(f.w for f in faces))), 0, 1)
            if faces else skin_tone)

    # exposure first (subject then background), so later ops work on corrected tones
    img = _expose_subject(img, alpha, subject_ev)
    img = _expose_background(img, alpha, background_ev)
    if wants_auto:
        img = _auto_levels(img, skin, alpha, faces, skin_target, subject_target,
                           bg_target, CONFIG.auto_levels_max_ev)

    hair = None
    if faces:
        face_w = max(f.w for f in faces)
        # geometry first (a warp changes where everything is), then tone/color ops
        img = _reduce_dewlap(img, faces, rp.get("reduce_dewlap", 0) * mult)
        # recover blown skin before softening/tone so later ops work on corrected tones
        img = _tame_skin_highlights(img, skin, face_w, rp.get("tame_highlights", 0) * mult)
        # blemish erase next, so skin softening blends over the cloned patches
        if rp.get("remove_blemishes", False):
            img = _remove_blemishes(img, skin, faces, 0.8 * mult)
        img = _soften_skin(img, skin, rp.get("skin_smoothing", 0) * mult, face_w)
        img = _skin_tone_correction(img, skin_tone, rp.get("skin_tone_correction", 0) * mult,
                                    rp.get("skin_type", "auto"),
                                    rp.get("skin_warmth", CONFIG.skin_orange_warmth),
                                    rp.get("skin_saturation", CONFIG.skin_saturation),
                                    rp.get("skin_red", CONFIG.skin_red))
        img = _brighten_skin(img, skin_tone, rp.get("skin_luminance", CONFIG.skin_luminance) * mult)
        img = _reduce_dark_circles(img, faces, rp.get("reduce_dark_circles", 0) * mult, skin)
        img = _brighten_eyes(img, faces, rp.get("brighten_eyes", 0) * mult)
        img = _iris_pop(img, faces, rp.get("iris_enhance", 0) * mult)
        img = _whiten_teeth(img, faces, rp.get("whiten_teeth", 0) * mult)
        img = _enhance_lips(img, faces, rp.get("lip_enhance", CONFIG.lip_enhance) * mult)
        if wants_hair or wants_cloth_grade:
            hair = _hair_mask(img, faces, alpha, skin)   # also lets clothing grade skip hair
        if wants_hair:
            img = _enhance_hair(img, hair, alpha,
                                rp.get("hair_texture", 0) * mult,
                                rp.get("hair_shimmer", 0) * mult,
                                rp.get("hair_defrizz", 0) * mult, face_w)

    img = _clothing_contrast(img, alpha, skin, rp.get("clothing_contrast", 0) * mult)
    if wants_cloth_grade:
        img = _clothing_color_grade(
            img, alpha, skin, hair, clothing.get("color", ""),
            float(clothing.get("color_pop", 0) or 0), float(clothing.get("luminance", 0) or 0),
            float(clothing.get("shadows", 0) or 0), float(clothing.get("blacks", 0) or 0),
            float(clothing.get("whites", 0) or 0), mult)
    img = _apply_background(img, bg_action, bg.get("replace_prompt", ""), alpha,
                           bg.get("color", "gray"))

    dest = Path(tiff_path).with_suffix(".retouched.tiff")
    _save_tiff(img, dest)
    log.info("Retouched %s -> %s (faces=%d, bg=%s)", tiff_path.name, dest.name,
             len(faces), bg_action)
    return dest
