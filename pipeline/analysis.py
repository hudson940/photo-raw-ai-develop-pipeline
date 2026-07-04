"""Stage 3: AI vision analysis -> schema-guaranteed develop + retouch parameters.

Sends the preview JPEG to Claude with structured outputs enabled, so the
response is validated against the PhotoAnalysis schema by the API itself.
Field descriptions below double as the model's instructions for each value.
"""

import base64
import io
import logging
import random
from pathlib import Path
from typing import Literal

import anthropic
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .config import CONFIG

log = logging.getLogger("analysis")


class VisionNotSupported(Exception):
    """The configured endpoint accepts image blocks but the model never sees them."""


def verify_vision(client: anthropic.Anthropic | None = None) -> None:
    """Canary check: some Anthropic-compatible gateways (e.g. text-only models behind
    a proxy) silently DROP image blocks, and the model then hallucinates an analysis.
    We render a random number into an image and require the model to read it back."""
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
        max_tokens=50,
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
    action: Literal["keep", "blur", "smooth", "studio", "replace"] = Field(
        description="What to do with the background: 'keep' (default, almost always right), "
        "'blur' soft bokeh when the background is busy/distracting, 'smooth' de-wrinkle a "
        "studio/paper/cloth backdrop that has visible folds, creases, lint or marks (keeps it "
        "reading as a flat backdrop), 'studio' replace with a neutral gray studio backdrop, "
        "'replace' generate a new background (needs replace_prompt)"
    )
    replace_prompt: str = Field(
        description="Short description of the new background when action='replace', "
        "e.g. 'soft window light, bright interior'. Empty string otherwise."
    )


class RetouchParams(BaseModel):
    is_portrait: bool = Field(description="True only if a human face is a clear subject of the image")
    skin_smoothing: float = Field(description="Skin smoothing strength 0.0-1.0 (frequency separation, preserves pores); keep <=0.5 for a natural look; 0 if not a portrait")
    remove_blemishes: bool = Field(description="Erase temporary blemishes — pimples, acne marks, skin tags, and stray hairs over skin — by cloning nearby skin; never removes moles or permanent identifying features")
    skin_tone_correction: float = Field(description="Even out blotchy/uneven skin tone and restore a healthy tone 0.0-1.0; 0.2-0.4 typical; 0 to leave tone untouched")
    reduce_dark_circles: float = Field(description="Lighten under-eye dark circles / eye bags 0.0-1.0; keep subtle (0.2-0.4); 0 if none")
    reduce_dewlap: float = Field(description="Subtly reduce a dewlap / double chin / sagging under-chin skin by nudging it upward 0.0-1.0; keep subtle (0.2-0.4); 0 if not needed")
    brighten_eyes: float = Field(description="Brighten the whites of the eyes (sclera) and reduce bloodshot redness 0.0-1.0, subtle values preferred")
    iris_enhance: float = Field(description="Add a subtle saturation/clarity pop to the irises 0.0-1.0; 0.2-0.4 typical; 0 if eyes not clearly visible")
    whiten_teeth: float = Field(description="Teeth whitening strength 0.0-1.0 (desaturates yellow + brightens); 0 if teeth not visible")
    clothing_contrast: float = Field(
        description="Local contrast/texture boost on clothing and fabric 0.0-1.0; "
        "0.2-0.4 gives cloth more definition; 0 if no clothing is prominent"
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
temporary spots or stray hairs on the skin.
- Background: 'keep' unless the background clearly hurts the photo. Use 'blur' for busy or \
cluttered backgrounds behind a portrait, 'smooth' when the subject stands against a studio/paper/\
cloth backdrop that shows wrinkles, folds or creases, 'studio' when a clean corporate/profile \
look fits, 'replace' only when the background is unsalvageable (describe the new one in replace_prompt).
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
    "reduce_dark_circles": 0.0,
    "reduce_dewlap": 0.0,
    "brighten_eyes": 0.0,
    "iris_enhance": 0.0,
    "whiten_teeth": 0.0,
    "clothing_contrast": 0.0,
    "background": {"action": "keep", "replace_prompt": ""},
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


def analyze_preview(preview_path: Path, client: anthropic.Anthropic | None = None) -> PhotoAnalysis:
    """Run vision analysis on a preview JPEG. Raises on API or validation failure."""
    client = client or anthropic.Anthropic()
    image_b64 = base64.standard_b64encode(preview_path.read_bytes()).decode()

    response = client.messages.create(
        model=CONFIG.model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
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
                {"type": "text", "text": USER_PROMPT},
            ],
        }],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Model refused to analyze this image")

    raw = "".join(b.text for b in response.content if b.type == "text")
    cleaned = _strip_fences(raw)
    result = PhotoAnalysis.model_validate_json(cleaned)

    log.info(
        "Analyzed %s: portrait=%s confidence=%.2f — %s",
        preview_path.name,
        result.retouch.is_portrait,
        result.confidence,
        result.scene_description,
    )
    return result
