"""Stage 3: AI vision analysis -> schema-guaranteed develop + retouch parameters.

Sends the preview JPEG to Claude with structured outputs enabled, so the
response is validated against the PhotoAnalysis schema by the API itself.
Field descriptions below double as the model's instructions for each value.
"""

import base64
import io
import json
import logging
import random
from pathlib import Path
from typing import Literal

import anthropic
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .config import CONFIG

log = logging.getLogger("analysis")


class VisionNotSupported(Exception):
    """The configured endpoint accepts image blocks but the model never sees them."""


def verify_vision(client: anthropic.Anthropic | None = None) -> None:
    """Canary check: some Anthropic-compatible gateways (e.g. text-only models behind
    a proxy) silently DROP image blocks, and the model then hallucinates an analysis.
    We render a random number into an image and require the model to read it back.

    The token budget must be generous: local reasoning models (e.g. Qwen3-VL via LM
    Studio) spend 100+ tokens "thinking" before answering, so a tiny budget gets
    exhausted mid-reasoning and returns empty content — which looks like a dropped
    image but isn't. CONFIG.canary_max_tokens leaves room for that reasoning."""
    client = client or anthropic.Anthropic()
    secret = f"{random.randint(100, 999)}"
    img = Image.new("RGB", (256, 128), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 40), secret, fill="black", font_size=48)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

    response = client.messages.create(
        model=CONFIG.model,
        max_tokens=CONFIG.canary_max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": "What number is written in this image? Reply with the digits only."},
            ],
        }],
    )
    reply = "".join(b.text for b in response.content if b.type == "text")
    if secret not in reply:
        raise VisionNotSupported(
            f"Vision canary failed: model '{CONFIG.model}' could not read the test image "
            f"(expected {secret!r}, got {reply.strip()!r}). The configured endpoint likely "
            "drops image blocks — Stage 3 analysis would hallucinate. Use a vision-capable "
            "model/endpoint (e.g. the Anthropic API with claude-opus-4-8)."
        )
    log.info("Vision canary passed for model %s", CONFIG.model)


class WhiteBalance(BaseModel):
    mode: Literal["camera", "kelvin"] = Field(
        description="'camera' = keep the camera's as-shot white balance (accurate, the default and "
        "correct choice for almost all photos). 'kelvin' = override with a specific color "
        "temperature; use only when there is an obvious color cast the camera got wrong."
    )
    temp: int = Field(description="Color temperature in Kelvin (2500-9000), used only when mode='kelvin'; higher = warmer, lower = cooler. Relative to the as-shot balance (6500 = no change)")
    tint: int = Field(description="Green-magenta tint correction, -50 (green) to +50 (magenta), 0 = neutral; applied only when mode='kelvin'")


class Crop(BaseModel):
    x: float = Field(description="Left edge of crop as fraction of width, 0.0-1.0")
    y: float = Field(description="Top edge of crop as fraction of height, 0.0-1.0")
    w: float = Field(description="Crop width as fraction of original width, 0.0-1.0")
    h: float = Field(description="Crop height as fraction of original height, 0.0-1.0")
    aspect: str = Field(description="Target aspect ratio, e.g. '3:2', '4:5', '1:1', or 'original'")


class DevelopParams(BaseModel):
    exposure_ev: float = Field(description="Exposure correction in EV stops, typically -2.0 to +2.0")
    white_balance: WhiteBalance
    contrast: float = Field(description="Contrast adjustment, -1.0 to +1.0, 0 = no change")
    highlights: float = Field(description="Highlight recovery, -100 (recover) to +100 (boost)")
    shadows: float = Field(description="Shadow lift, -100 (crush) to +100 (lift)")
    saturation: float = Field(description="Global saturation, -1.0 to +1.0, 0 = no change")
    vibrance: float = Field(description="Vibrance (protects skin tones), -1.0 to +1.0")
    crop: Crop
    rotation_deg: float = Field(description="Horizon straightening in degrees, positive = clockwise, usually within ±5")


class Background(BaseModel):
    action: Literal["auto", "keep", "blur", "smooth", "studio", "replace"] = Field(
        description="Background handling. ALWAYS output 'keep' — background changes are made "
        "manually by the operator, never chosen automatically. (The other modes exist for the "
        "CLI override: 'auto' picks smooth/studio/keep from the scene, 'blur' soft bokeh, "
        "'smooth' de-wrinkle a backdrop, 'studio' neutral backdrop, 'replace' generate a new one.)"
    )
    replace_prompt: str = Field(
        description="Short description of the new background when action='replace', "
        "e.g. 'soft window light, bright interior'. Empty string otherwise."
    )
    color: str = Field(
        description="Studio backdrop color when action='studio': a name (gray, white, black, "
        "charcoal, blue, navy, teal, green, red, maroon, pink, purple, beige, brown) or a #hex. "
        "Default 'gray'."
    )


class Clothing(BaseModel):
    color: str = Field(
        description="Dominant color of the subject's most prominent clothing, as a simple name: "
        "black, white, gray, red, orange, yellow, green, teal, blue, navy, purple, pink, brown, "
        "beige; 'multicolor' for prints/patterns; 'none' if no clothing is prominent. Used to "
        "pick the right treatment (neutral fabrics are graded but not saturated)."
    )
    color_pop: float = Field(
        description="Release/pop the fabric color: vibrance-weighted saturation boost 0.0-1.0. "
        "Use 0.2-0.4 for a colorful garment to make it richer; keep 0 for black/white/gray fabric "
        "(saturating a neutral just adds noise and color casts)."
    )
    luminance: float = Field(
        description="Brighten (+) or darken (-) the clothing, -1.0 to 1.0, 0 = no change. Keep "
        "small (±0.2); raise slightly for dark, muddy clothing, lower if the garment is too bright."
    )
    shadows: float = Field(
        description="Lift (+) or deepen (-) the shadow areas of the clothing, -1.0 to 1.0. Deepen "
        "slightly (-0.2) for richer fabric with more depth; lift to open up detail in dark folds."
    )
    blacks: float = Field(
        description="Black point of the clothing, -1.0 to 1.0. A small negative (-0.2 to -0.3) "
        "gives rich, deep blacks — ideal for dark or black garments (suits, black dresses)."
    )
    whites: float = Field(
        description="White point of the clothing, -1.0 to 1.0. Small positive brightens fabric "
        "highlights; use a small negative for white/light garments (wedding dress, white shirt) "
        "so the whites stay clean and do not blow out to detail-less white."
    )


class RetouchParams(BaseModel):
    is_portrait: bool = Field(description="True only if a human face is a clear subject of the image")
    skin_smoothing: float = Field(description="Skin smoothing strength 0.0-1.0 (frequency separation, preserves pores); keep <=0.5 for a natural look; 0 if not a portrait")
    remove_blemishes: bool = Field(description="Erase temporary blemishes — pimples, acne marks, skin tags, and stray hairs over skin — by cloning nearby skin; never removes moles or permanent identifying features")
    skin_tone_correction: float = Field(description="Even out blotchy/uneven skin tone and restore a healthy tone 0.0-1.0; 0.2-0.4 typical; 0 to leave tone untouched")
    skin_type: Literal["auto", "fair", "light", "medium", "olive", "tan", "brown", "deep"] = Field(
        description="Subject's skin type/tone, so tone correction targets the RIGHT healthy color "
        "for that skin (fair skin should not be pushed orange; deep skin should stay a rich warm "
        "brown, not sunburnt-red). fair=very light/pale, light=light, medium=light-brown, "
        "olive=medium with a green/sallow undertone, tan=tanned/olive-brown, brown=brown, "
        "deep=dark/deep brown. Use 'auto' to let the pipeline measure it. 'auto' if not a portrait."
    )
    reduce_dark_circles: float = Field(description="Lighten under-eye dark circles / eye bags 0.0-1.0; keep subtle (0.2-0.4); 0 if none")
    reduce_dewlap: float = Field(description="Subtly reduce a dewlap / double chin / sagging under-chin skin by nudging it upward 0.0-1.0; keep subtle (0.2-0.4); 0 if not needed")
    brighten_eyes: float = Field(description="Brighten the whites of the eyes (sclera) and reduce bloodshot redness 0.0-1.0, subtle values preferred")
    iris_enhance: float = Field(description="Add a subtle saturation/clarity pop to the irises 0.0-1.0; 0.2-0.4 typical; 0 if eyes not clearly visible")
    whiten_teeth: float = Field(description="Teeth whitening strength 0.0-1.0 (desaturates yellow + brightens); 0 if teeth not visible")
    lip_enhance: float = Field(description="Lip enhancement 0.0-1.0: restore/deepen natural lip red + a little richness and definition; 0.3-0.4 typical for portraits, 0 if lips not visible")
    tame_highlights: float = Field(description="Recover overexposed / shiny skin 0.0-1.0: pulls blown bright skin patches back toward the surrounding skin tone; 0.3-0.5 for hotspots, 0 if the skin is well exposed")
    hair_texture: float = Field(description="Hair texture & clarity to make individual strands pop 0.0-1.0; ON BY DEFAULT for portraits with visible hair (use 0.4-0.6); 0 if hair is not visible / covered / bald")
    hair_shimmer: float = Field(description="Add tonal contrast to hair (lift highlights, deepen shadows) for a natural shimmer 0.0-1.0; optional, 0 by default")
    hair_defrizz: float = Field(description="Reduce frizz / flyaway strands for a smoother, straighter hair look 0.0-1.0; optional, 0 by default")
    clothing_contrast: float = Field(
        description="Local contrast/texture boost on clothing and fabric 0.0-1.0; "
        "0.2-0.4 gives cloth more definition; 0 if no clothing is prominent"
    )
    clothing: Clothing
    subject_exposure: float = Field(
        description="Brighten (or darken) ONLY the subject/person, in EV stops (-1.0 to +2.0, "
        "0 = no change); uses the subject mask so the background is untouched. Raise it when "
        "the subject is underexposed relative to the background (e.g. backlit)."
    )
    background_exposure: float = Field(
        description="Brighten (or darken) ONLY the background, in EV stops (-2.0 to +1.0, "
        "0 = no change); uses the subject mask so the subject is untouched. Lower it to darken "
        "a distracting/overexposed background and make the subject stand out."
    )
    background: Background
    intensity: Literal["subtle", "natural", "polished"] = Field(
        description="Overall retouch strength: 'subtle' barely visible, 'natural' default, 'polished' editorial"
    )


class PhotoAnalysis(BaseModel):
    scene_description: str = Field(description="One sentence: what the photo shows and its main technical issues")
    develop: DevelopParams
    retouch: RetouchParams
    confidence: float = Field(description="Your confidence 0.0-1.0 that these parameters will improve the photo; lower it for ambiguous, badly damaged, or unusual images")


SYSTEM_PROMPT = """You are an expert photo editor and retoucher analyzing photographs for an \
automated RAW development pipeline. You examine a preview image and prescribe conservative, \
tasteful correction parameters.

Principles:
- Correct toward a neutral, professionally exposed and color-balanced result. Do not stylize.
- White balance: default to mode='camera' (keep the camera's accurate as-shot balance). Only \
use mode='kelvin' when there is a clear, wrong color cast; then pick a temp (higher = warmer).
- Prefer small corrections; most photos need modest adjustments, not dramatic ones.
- Only crop to fix clear composition problems (distracting edges, badly off-center subject); \
otherwise keep the full frame (crop x=0, y=0, w=1, h=1, aspect='original').
- For portraits, retouching must stay natural: preserve skin texture, moles, and identity. \
Flag only genuinely temporary flaws. Typical good values: skin_smoothing 0.2-0.4, \
skin_tone_correction 0.2-0.4, reduce_dark_circles 0.2-0.4, reduce_dewlap 0.2-0.4 (only if there \
is a visible double chin / sagging under-chin skin), brighten_eyes 0.1-0.3, iris_enhance 0.2-0.4, \
whiten_teeth 0.1-0.3, clothing_contrast 0.2-0.4. Set remove_blemishes true when you can see \
temporary spots or stray hairs on the skin. Set tame_highlights 0.3-0.5 when parts of the skin \
(forehead, nose, cheeks) look blown out / shiny white; 0 otherwise.
- Clothing: identify the dominant clothing color (clothing.color) and grade the fabric like a \
photo editor would. For a COLORFUL garment, release the color with clothing.color_pop 0.2-0.4 and \
optionally deepen clothing.shadows -0.1 to -0.2 for richness. For a BLACK/dark garment, set \
clothing.color_pop 0 and clothing.blacks -0.2 to -0.3 for deep, rich blacks. For a WHITE/light \
garment, set clothing.color_pop 0 and clothing.whites -0.1 to -0.2 so it stays clean and doesn't \
blow out. Keep clothing.luminance near 0 unless the garment is clearly too dark or too bright. \
Set clothing.color='none' and all clothing values 0 when no clothing is prominent.
- Skin type: for portraits, set skin_type to the subject's actual skin tone (fair/light/medium/\
olive/tan/brown/deep) so tone correction targets the right healthy color — this is what keeps \
fair skin from going orange and deep skin from going red/ashy. Look carefully at the person's \
skin, not the lighting; pick 'olive' when a medium skin has a green/sallow undertone. Use 'auto' \
only if you truly cannot tell. Set skin_type='auto' when the image is not a portrait.
- Hair: by default enhance hair_texture (0.4-0.6) whenever hair is visible, to make strands \
pop; set it to 0 only if hair is not visible, covered, or the subject is bald. hair_shimmer \
and hair_defrizz are optional extras (leave at 0 unless the hair clearly benefits).
- subject_exposure: leave at 0 normally. Raise it (+0.3 to +1.0 EV) only when the subject is \
clearly underexposed relative to the background, e.g. backlit or in shadow against a bright scene.
- background_exposure: leave at 0 normally. Lower it (-0.3 to -1.0 EV) when the background is \
distracting or overexposed and darkening it would help the subject stand out.
- highlights/shadows (develop block): recover blown highlights with negative highlights, and \
lift dark shadows with positive shadows; keep both modest unless the photo clearly needs it.
- Background: ALWAYS set background.action = 'keep'. Do NOT change the background on your own — \
blur/smooth/studio/replace are manual choices made by the operator via a command-line flag, not \
something you decide. Leave replace_prompt empty.
- If the image is not a portrait, set is_portrait=false and zero out all retouch strengths.
- Be honest in the confidence score: dark/blurry/unreadable previews or unusual subjects \
warrant low confidence so a human reviews them."""

def _json_schema_hint() -> str:
    """Return a compact JSON example matching the PhotoAnalysis schema for the system prompt."""
    return """\
{
  "scene_description": "Brief description of the photo and its main technical issues",
  "develop": {
    "exposure_ev": 0.0,
    "white_balance": {"mode": "camera", "temp": 6500, "tint": 0},
    "contrast": 0.0,
    "highlights": 0.0,
    "shadows": 0.0,
    "saturation": 0.0,
    "vibrance": 0.0,
    "crop": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "aspect": "original"},
    "rotation_deg": 0.0
  },
  "retouch": {
    "is_portrait": false,
    "skin_smoothing": 0.0,
    "remove_blemishes": false,
    "skin_tone_correction": 0.0,
    "skin_type": "auto",
    "reduce_dark_circles": 0.0,
    "reduce_dewlap": 0.0,
    "brighten_eyes": 0.0,
    "iris_enhance": 0.0,
    "whiten_teeth": 0.0,
    "lip_enhance": 0.35,
    "tame_highlights": 0.0,
    "hair_texture": 0.5,
    "hair_shimmer": 0.0,
    "hair_defrizz": 0.0,
    "clothing_contrast": 0.0,
    "clothing": {"color": "none", "color_pop": 0.0, "luminance": 0.0, "shadows": 0.0, "blacks": 0.0, "whites": 0.0},
    "subject_exposure": 0.0,
    "background_exposure": 0.0,
    "background": {"action": "keep", "replace_prompt": "", "color": "gray"},
    "intensity": "natural"
  },
  "confidence": 0.8
}"""


USER_PROMPT = "Analyze this photograph and prescribe develop and retouch parameters. Return ONLY valid JSON matching the schema from the system prompt, with no additional text."


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from an LLM response."""
    t = text.strip()
    if t.startswith("```"):
        t = t[t.index("\n") + 1:] if "\n" in t else t[3:]
    if t.endswith("```"):
        t = t[:t.rindex("```")].rstrip()
    return t


def _balanced_objects(t: str) -> list[str]:
    """Return every top-level {...} object in a string, respecting quoted strings."""
    out, i, n = [], 0, len(t)
    while i < n:
        if t[i] == "{":
            depth, in_str, esc, j = 0, False, False, i
            while j < n:
                c = t[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(t[i:j + 1])
                        i = j
                        break
                j += 1
        i += 1
    return out


def _extract_json(text: str) -> str:
    """Pull the JSON object out of an LLM reply, tolerating a reasoning preamble or
    trailing prose (some models, e.g. Gemma, emit their thoughts as plain text before
    the JSON instead of in a separate field)."""
    t = _strip_fences(text)
    try:
        json.loads(t)
        return t                      # already clean JSON
    except ValueError:
        pass
    candidates = _balanced_objects(t)
    # Prefer an object that looks like our schema; otherwise the largest one.
    schema_like = [c for c in candidates if '"develop"' in c or '"scene_description"' in c]
    pool = schema_like or candidates
    return max(pool, key=len) if pool else t


_THINKING_BUDGET = {"low": 1024, "medium": 2048, "high": 6000}
_EFFORT_OFF = {"off", "none", "no", "disabled", "0"}
# Send no reasoning field at all — for models/servers that don't support reasoning control
# (e.g. GGUFs with no reasoning KVs, which otherwise log "cannot be converted to custom KVs").
_EFFORT_PASSTHROUGH = {"", "default", "model", "passthrough", "auto"}


def _reasoning_kwargs(client: anthropic.Anthropic) -> dict:
    """Translate CONFIG.reasoning_effort into request kwargs for the active backend.

    The real Anthropic API uses a `thinking` token budget; local OpenAI/Anthropic-compatible
    servers (LM Studio, etc.) use OpenAI's `reasoning_effort`. Sending `reasoning_effort` to
    the real API would error, so we pick based on the base URL. Use 'default' to send nothing
    (the right choice for local models that don't expose any reasoning control).
    """
    eff = (CONFIG.reasoning_effort or "").strip().lower()
    if eff in _EFFORT_PASSTHROUGH:
        return {}
    is_anthropic = "anthropic.com" in str(getattr(client, "base_url", ""))
    if is_anthropic:
        if eff in _EFFORT_OFF:
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "enabled", "budget_tokens": _THINKING_BUDGET.get(eff, 1024)}}
    # local server: no Anthropic thinking param; steer the model via reasoning_effort
    return {"extra_body": {"reasoning_effort": "low" if eff in _EFFORT_OFF else eff}}


def _subject_only_jpeg(preview_path: Path) -> bytes:
    """Return JPEG bytes of the preview with the background replaced by neutral gray, so the
    model meters exposure/white balance/color on the subject. Falls back to the full frame
    when no usable subject mask is found."""
    from .retouch import _subject_alpha  # lazy: pulls in cv2/rembg only when needed

    img = np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.float32) / 255.0
    alpha = _subject_alpha(img)
    if alpha is None or alpha.max() < 0.05 or alpha.mean() > 0.95:
        log.warning("subject-only analysis: no usable subject mask — using the full frame")
        return preview_path.read_bytes()
    a = alpha[..., None]
    out = np.clip(img * a + 0.5 * (1 - a), 0, 1)   # neutral mid-gray background
    buf = io.BytesIO()
    Image.fromarray((out * 255).astype(np.uint8)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _skin_only_jpeg(preview_path: Path) -> bytes:
    """Return JPEG bytes of the preview with everything except skin masked to neutral gray, so
    the model meters exposure on the skin. Uses an orientation-independent color detector (the
    preview is often sideways, where face detection fails). Falls back to the full frame when
    little skin is found."""
    import cv2

    img = np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    ycrcb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    skin = (((cr >= 133) & (cr <= 176) & (cb >= 77) & (cb <= 127) & (y > 40))
            .astype(np.uint8) * 255)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drop speckle
    skinf = cv2.GaussianBlur(skin.astype(np.float32) / 255.0, (0, 0), max(h, w) * 0.004)
    if skinf.max() < 0.05 or skinf.mean() < 0.004:
        log.warning("skin-exposure: little skin detected — using the full frame")
        return preview_path.read_bytes()
    a = np.clip(skinf, 0, 1)[..., None]
    out = np.clip(img * a + 0.5 * (1 - a), 0, 1)   # keep skin, gray everywhere else
    buf = io.BytesIO()
    Image.fromarray((out * 255).astype(np.uint8)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def analyze_preview(preview_path: Path, client: anthropic.Anthropic | None = None) -> PhotoAnalysis:
    """Run vision analysis on a preview JPEG. Raises on API or validation failure."""
    client = client or anthropic.Anthropic()

    user_prompt = USER_PROMPT
    if CONFIG.analyze_skin_exposure:
        image_bytes = _skin_only_jpeg(preview_path)
        user_prompt += ("\n\nOnly the subject's SKIN is shown (everything else is masked gray). "
                        "Meter the exposure on the skin: set exposure_ev (and highlights/shadows) "
                        "so the skin is well exposed and NOT blown out — if the skin already looks "
                        "bright, use a low or negative exposure_ev. Keep other values reasonable.")
    elif CONFIG.analyze_subject_only:
        image_bytes = _subject_only_jpeg(preview_path)
        user_prompt += ("\n\nThe background has been masked to neutral gray on purpose: base "
                        "exposure, white balance, and all color/tone decisions on the SUBJECT "
                        "only, ignoring the gray area.")
    else:
        image_bytes = preview_path.read_bytes()
    image_b64 = base64.standard_b64encode(image_bytes).decode()

    response = client.messages.create(
        model=CONFIG.model,
        max_tokens=16000,
        **_reasoning_kwargs(client),
        system=SYSTEM_PROMPT + "\n\nOutput ONLY valid JSON matching this exact schema (no markdown fences, no extra text):\n" + _json_schema_hint(),
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        }],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Model refused to analyze this image")

    raw = "".join(b.text for b in response.content if b.type == "text")
    cleaned = _extract_json(raw)
    try:
        result = PhotoAnalysis.model_validate_json(cleaned)
    except ValueError as exc:
        snippet = raw.strip()[:300].replace("\n", " ")
        raise RuntimeError(
            f"Model '{CONFIG.model}' did not return valid analysis JSON. "
            f"Response started with: {snippet!r}"
        ) from exc

    log.info(
        "Analyzed %s: portrait=%s confidence=%.2f — %s",
        preview_path.name,
        result.retouch.is_portrait,
        result.confidence,
        result.scene_description,
    )
    return result
