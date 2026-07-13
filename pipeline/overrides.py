"""Shared retouch / white-balance override flags.

The same set of `--skin`, `--dewlap`, `--wb`, ... flags are accepted by both
`python -m pipeline.redo` (one photo, from the queue) and `python -m pipeline.run`
(every photo the worker processes). This module defines the flags once and turns
them into an overrides dict that is merged onto a photo's analysis before
develop/retouch.
"""

import argparse

# retouch flag -> (dest, kind); kind "float" | "bool"; float flags also imply a portrait
_RETOUCH_FLOAT_FLAGS = {
    "skin": "skin_smoothing",
    "skintone": "skin_tone_correction",
    "darkcircles": "reduce_dark_circles",
    "dewlap": "reduce_dewlap",
    "eyes": "brighten_eyes",
    "iris": "iris_enhance",
    "teeth": "whiten_teeth",
    "lips": "lip_enhance",
    "tame_highlights": "tame_highlights",   # recover overexposed / shiny skin
    "hair": "hair_texture",
    "hair_shimmer": "hair_shimmer",
    "defrizz": "hair_defrizz",
    "clothing": "clothing_contrast",       # not face-specific, but still a retouch param
    "subject_exposure": "subject_exposure",        # EV stops, brightens only the subject
    "background_exposure": "background_exposure",   # EV stops, brightens only the background
}
_FACE_FLAGS = {"skin", "skintone", "darkcircles", "dewlap", "eyes", "iris", "teeth", "lips",
               "tame_highlights", "hair", "hair_shimmer", "defrizz"}


def add_override_args(parser: argparse.ArgumentParser) -> None:
    """Register the retouch/white-balance override flags on an argument parser."""
    g = parser.add_argument_group("retouch overrides (applied on top of the AI analysis)")
    g.add_argument("--skin", type=float, metavar="0..1", help="skin smoothing strength")
    g.add_argument("--blemishes", dest="blemishes", action="store_true", default=None,
                   help="erase pimples, acne marks, skin tags, stray hairs")
    g.add_argument("--no-blemishes", dest="blemishes", action="store_false",
                   help="disable blemish removal")
    g.add_argument("--skintone", type=float, metavar="0..1",
                   help="even out skin tone / healthy color correction")
    g.add_argument("--skin-type", dest="skin_type",
                   choices=["auto", "fair", "light", "medium", "olive", "tan", "brown", "deep"],
                   help="skin tone to target for tone correction (auto = measure it); "
                        "picks the right healthy color per skin type. Implies a default --skintone")
    g.add_argument("--skin-warmth", dest="skin_warmth", type=float, metavar="-1..1",
                   help="bias skin tone toward orange (+) or yellow (-); raise if skin looks too "
                        "yellow. Implies a default --skintone")
    g.add_argument("--skin-saturation", dest="skin_saturation", type=float, metavar="0.5..1.8",
                   help="skin richness (1.0 = as corrected, >1 = deeper/richer color, <1 = muted). "
                        "Implies a default --skintone")
    g.add_argument("--skin-luminance", dest="skin_luminance", type=float, metavar="0..1",
                   help="brighten skin (extra on lips/cheeks), like the Lightroom Orange/Red "
                        "luminance sliders; 0 = no brightening")
    g.add_argument("--skin-red", dest="skin_red", type=float, metavar="0..1",
                   help="add a healthy red flush to the skin midtones/cheeks; 0 = none")
    g.add_argument("--darkcircles", type=float, metavar="0..1",
                   help="reduce under-eye dark circles / eye bags")
    g.add_argument("--dewlap", type=float, metavar="0..1",
                   help="reduce a dewlap / double chin (subtle under-chin lift)")
    g.add_argument("--eyes", type=float, metavar="0..1", help="eye-white (sclera) brightening")
    g.add_argument("--iris", type=float, metavar="0..1", help="iris saturation/clarity pop")
    g.add_argument("--teeth", type=float, metavar="0..1", help="teeth whitening strength")
    g.add_argument("--lips", type=float, metavar="0..1",
                   help="lip enhancement: restore/deepen natural lip red + richness and definition")
    g.add_argument("--tame-highlights", type=float, metavar="0..1",
                   help="recover overexposed / shiny skin (pull blown areas toward surrounding tone)")
    g.add_argument("--hair", type=float, metavar="0..1",
                   help="hair texture & clarity — make strands pop (on by default for portraits)")
    g.add_argument("--no-hair", action="store_true",
                   help="disable hair texture/clarity enhancement")
    g.add_argument("--hair-shimmer", type=float, metavar="0..1",
                   help="hair shimmer: raise highlights, drop shadows")
    g.add_argument("--defrizz", type=float, metavar="0..1",
                   help="reduce frizz / flyaways (smoother, straighter hair)")
    g.add_argument("--clothing", type=float, metavar="0..1", help="clothing contrast strength")
    g.add_argument("--cloth-color", dest="cloth_color", metavar="NAME",
                   help="dominant clothing color (black/white/red/blue/...); neutrals aren't saturated")
    g.add_argument("--cloth-pop", dest="cloth_pop", type=float, metavar="0..1",
                   help="release/pop the clothing color (vibrance-weighted saturation)")
    g.add_argument("--cloth-luminance", dest="cloth_luminance", type=float, metavar="-1..1",
                   help="brighten (+) / darken (-) the clothing")
    g.add_argument("--cloth-shadows", dest="cloth_shadows", type=float, metavar="-1..1",
                   help="lift (+) / deepen (-) clothing shadows")
    g.add_argument("--cloth-blacks", dest="cloth_blacks", type=float, metavar="-1..1",
                   help="clothing black point (- = rich deep blacks, great for dark garments)")
    g.add_argument("--cloth-whites", dest="cloth_whites", type=float, metavar="-1..1",
                   help="clothing white point (- = protect/recover, great for white garments)")
    g.add_argument("--subject-exposure", type=float, metavar="EV",
                   help="brighten (+) or darken (-) only the subject, in EV stops (e.g. 0.5, -0.3)")
    g.add_argument("--background-exposure", type=float, metavar="EV",
                   help="brighten (+) or darken (-) only the background, in EV stops")
    g.add_argument("--shadows", type=float, metavar="-100..100",
                   help="lift (+) or crush (-) shadows globally")
    g.add_argument("--highlights", type=float, metavar="-100..100",
                   help="boost (+) or recover (-) highlights globally")
    g.add_argument("--background", choices=["auto", "keep", "blur", "smooth", "studio", "replace"],
                   help="background handling (auto = pick smooth/studio/keep from the scene; "
                        "smooth = de-wrinkle a backdrop)")
    g.add_argument("--studio-color", metavar="NAME|#hex",
                   help="studio backdrop color: gray/white/black/charcoal/blue/navy/teal/green/"
                        "red/maroon/pink/purple/beige/brown or a #hex; implies --background studio")
    g.add_argument("--prompt", help="background description for --background replace")
    g.add_argument("--erase-mask", dest="erase_mask", metavar="PNG",
                   help="erase the white regions of this mask image from the photo "
                        "(object removal; the web UI draws these masks for you)")
    g.add_argument("--erase-method", dest="erase_method",
                   choices=["content-aware", "generative"],
                   help="fill for erased regions: content-aware (deterministic, fast) or "
                        "generative (ComfyUI Generative Fill; needs ComfyUI running)")
    g.add_argument("--erase-prompt", dest="erase_prompt", metavar="TEXT",
                   help="what the generative fill should paint in the erased area")
    g.add_argument("--clear-erase", dest="clear_erase", action="store_true",
                   help="remove a previously saved erase mask from this photo")
    g.add_argument("--wb", metavar="camera|KELVIN",
                   help="white balance: 'camera' (accurate as-shot) or a Kelvin number "
                        "like 5500 (higher = warmer)")
    g.add_argument("--tint", type=int, metavar="-50..50",
                   help="green-magenta tint for --wb kelvin mode (-50 green .. +50 magenta)")
    g.add_argument("--intensity", choices=["subtle", "natural", "polished"],
                   help="overall retouch strength")
    g.add_argument("--auto-levels", action="store_true", default=None,
                   help="per-layer auto exposure: meter face-skin / subject / background each "
                        "toward a midtone target and correct through their masks (deterministic, "
                        "no AI). Fixes over/under-exposed skin without touching the rest of the frame")
    g.add_argument("--skin-target", type=float, metavar="0..1",
                   help="target midtone luminance for face skin (implies --auto-levels; default 0.72)")
    g.add_argument("--subject-target", type=float, metavar="0..1",
                   help="target midtone for the subject's body/clothing (implies --auto-levels; default 0.5)")
    g.add_argument("--background-target", type=float, metavar="0..1",
                   help="target midtone for the background (implies --auto-levels; off by default)")
    g.add_argument("--subject-only", action="store_true",
                   help="when the AI computes automatic exposure/white-balance/color, meter on "
                        "the subject only (mask the background); applies at analysis time")
    g.add_argument("--skin-exposure", action="store_true",
                   help="meter the AI's auto exposure on the SKIN only (mask everything else) to "
                        "avoid over-exposing skin; applies at analysis time")


def overrides_from_args(args: argparse.Namespace) -> dict | None:
    """Collect the provided flags into {"retouch": {...}, "white_balance": {...}}.

    Only keys the user actually passed are included. Returns None if nothing was set.
    Raises SystemExit via argparse-style error for a malformed --wb value.
    """
    retouch: dict = {}
    for flag, key in _RETOUCH_FLOAT_FLAGS.items():
        val = getattr(args, flag, None)
        if val is not None:
            retouch[key] = val
            if flag in _FACE_FLAGS:
                retouch["is_portrait"] = True

    if getattr(args, "blemishes", None) is not None:
        retouch["remove_blemishes"] = args.blemishes
        retouch["is_portrait"] = True
    if getattr(args, "skin_type", None) is not None:
        retouch["skin_type"] = args.skin_type
        retouch["is_portrait"] = True
        retouch.setdefault("skin_tone_correction", 0.35)   # a type with no strength does nothing
    if getattr(args, "skin_warmth", None) is not None:
        retouch["skin_warmth"] = args.skin_warmth
        retouch["is_portrait"] = True
        retouch.setdefault("skin_tone_correction", 0.35)   # warmth needs tone correction active
    if getattr(args, "skin_saturation", None) is not None:
        retouch["skin_saturation"] = args.skin_saturation
        retouch["is_portrait"] = True
        retouch.setdefault("skin_tone_correction", 0.35)   # richness needs tone correction active
    if getattr(args, "skin_luminance", None) is not None:
        retouch["skin_luminance"] = args.skin_luminance
        retouch["is_portrait"] = True
    if getattr(args, "skin_red", None) is not None:
        retouch["skin_red"] = args.skin_red
        retouch["is_portrait"] = True
        retouch.setdefault("skin_tone_correction", 0.35)   # red flush rides on tone correction
    if getattr(args, "no_hair", False):
        retouch["hair_texture"] = 0.0

    # clothing color grade (nested, like background)
    clothing: dict = {}
    for flag, key in (("cloth_color", "color"), ("cloth_pop", "color_pop"),
                      ("cloth_luminance", "luminance"), ("cloth_shadows", "shadows"),
                      ("cloth_blacks", "blacks"), ("cloth_whites", "whites")):
        val = getattr(args, flag, None)
        if val is not None:
            clothing[key] = val
    if clothing:
        retouch["clothing"] = clothing

    # erase (nested, like background): a mask implies the op; --clear-erase wipes it
    erase: dict = {}
    if getattr(args, "clear_erase", False):
        erase["mask"] = ""
    elif getattr(args, "erase_mask", None) is not None:
        erase["mask"] = args.erase_mask
    if getattr(args, "erase_method", None) is not None:
        erase["method"] = args.erase_method
    if getattr(args, "erase_prompt", None) is not None:
        erase["prompt"] = args.erase_prompt
    if erase:
        retouch["erase"] = erase

    # auto-levels: any of the target flags implies the pass is on
    for flag, key in (("skin_target", "auto_skin_target"),
                      ("subject_target", "auto_subject_target"),
                      ("background_target", "auto_background_target")):
        val = getattr(args, flag, None)
        if val is not None:
            retouch[key] = val
            retouch["auto_levels"] = True
    if getattr(args, "auto_levels", None):
        retouch["auto_levels"] = True
    if getattr(args, "intensity", None) is not None:
        retouch["intensity"] = args.intensity
    if getattr(args, "background", None) is not None:
        retouch.setdefault("background", {})["action"] = args.background
    if getattr(args, "prompt", None) is not None:
        retouch.setdefault("background", {})["replace_prompt"] = args.prompt
    if getattr(args, "studio_color", None) is not None:
        b = retouch.setdefault("background", {})
        b["color"] = args.studio_color
        b.setdefault("action", "studio")   # picking a color implies a studio backdrop

    white_balance: dict = {}
    if getattr(args, "wb", None) is not None:
        if args.wb.lower() == "camera":
            white_balance["mode"] = "camera"
        else:
            try:
                white_balance.update(mode="kelvin", temp=int(args.wb))
            except ValueError:
                raise SystemExit(f"--wb must be 'camera' or a Kelvin number, got {args.wb!r}")
    if getattr(args, "tint", None) is not None:
        white_balance.update(mode="kelvin", tint=args.tint)

    # develop-stage tonal controls (global); changing these forces a re-develop
    develop: dict = {}
    if getattr(args, "shadows", None) is not None:
        develop["shadows"] = args.shadows
    if getattr(args, "highlights", None) is not None:
        develop["highlights"] = args.highlights

    ov: dict = {}
    if retouch:
        ov["retouch"] = retouch
    if white_balance:
        ov["white_balance"] = white_balance
    if develop:
        ov["develop"] = develop
    return ov or None


def apply_overrides(analysis: dict, overrides: dict | None) -> dict:
    """Merge overrides onto an analysis dict in place (retouch params + white balance)."""
    if not overrides:
        return analysis
    if "retouch" in overrides:
        rp = analysis.setdefault("retouch", {})
        for key, val in overrides["retouch"].items():
            if key in ("background", "clothing", "erase"):
                rp.setdefault(key, {}).update(val)   # nested groups merge, not replace
            else:
                rp[key] = val
    if "white_balance" in overrides:
        wb = analysis.setdefault("develop", {}).setdefault("white_balance", {})
        wb.update(overrides["white_balance"])
    if "develop" in overrides:
        analysis.setdefault("develop", {}).update(overrides["develop"])
    return analysis


def develop_overridden(overrides: dict | None) -> bool:
    """True if the run changed any develop-stage value (white balance, shadows, highlights),
    which means the RAW must be re-developed."""
    return bool(overrides and ("white_balance" in overrides or "develop" in overrides))


def background_overridden(overrides: dict | None) -> bool:
    """True if the run explicitly set a background action via --background."""
    return bool(overrides and overrides.get("retouch", {}).get("background", {}).get("action"))


def force_keep_background(analysis: dict) -> None:
    """Reset the background to 'keep'. Background is only ever changed on explicit request,
    so unless --background was passed we never touch it (even if the analysis says otherwise)."""
    bg = analysis.setdefault("retouch", {}).setdefault("background", {})
    bg["action"] = "keep"
    bg.setdefault("replace_prompt", "")


def describe(overrides: dict | None) -> str:
    """Compact one-line summary of active overrides, for logging."""
    if not overrides:
        return ""
    parts = []
    for k, v in overrides.get("retouch", {}).items():
        parts.append(f"{k}={v}")
    for k, v in overrides.get("develop", {}).items():
        parts.append(f"{k}={v}")
    for k, v in overrides.get("white_balance", {}).items():
        parts.append(f"wb.{k}={v}")
    return ", ".join(parts)
