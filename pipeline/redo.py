"""Re-run retouch on a processed photo with your own choices.

The AI proposes retouch parameters in Stage 3; this CLI lets you override any
of them per photo and re-render without re-running analysis or development:

    python -m pipeline.redo --list
    python -m pipeline.redo 3 --show
    python -m pipeline.redo 3 --skin 0.3 --background blur
    python -m pipeline.redo 3 --background replace --prompt "soft window light, bright interior"
    python -m pipeline.redo 3 --clothing 0.4 --eyes 0.2 --intensity subtle

The id argument also accepts a range or list, applying the same options to each:

    python -m pipeline.redo 91-98 --background studio --studio-color blue
    python -m pipeline.redo 91,93,95 --skin 0.3

Overrides are saved back to the queue database, so the decision log and any
later re-render keep your choices.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import CONFIG
from . import db
from .develop import develop
from .output import cleanup_intermediates, publish_done, write_rapidraw_sidecar
from .overrides import (
    add_override_args, apply_overrides, background_overridden,
    develop_overridden, force_keep_background, overrides_from_args,
)
from .retouch import retouch

log = logging.getLogger("redo")

_DEFAULT_RETOUCH = {
    "is_portrait": True,
    "skin_smoothing": 0.0,
    "remove_blemishes": False,
    "skin_tone_correction": 0.0,
    "skin_type": "auto",
    "skin_warmth": CONFIG.skin_orange_warmth,
    "skin_saturation": CONFIG.skin_saturation,
    "skin_luminance": CONFIG.skin_luminance,
    "skin_red": CONFIG.skin_red,
    "reduce_dark_circles": 0.0,
    "reduce_dewlap": 0.0,
    "brighten_eyes": 0.0,
    "iris_enhance": 0.0,
    "whiten_teeth": 0.0,
    "lip_enhance": CONFIG.lip_enhance,
    "tame_highlights": 0.0,
    "hair_texture": 0.5,
    "hair_shimmer": 0.0,
    "hair_defrizz": 0.0,
    "clothing_contrast": 0.0,
    "clothing": {"color": "none", "color_pop": 0.0, "luminance": 0.0,
                 "shadows": 0.0, "blacks": 0.0, "whites": 0.0},
    "subject_exposure": 0.0,
    "background_exposure": 0.0,
    "auto_levels": False,
    "background": {"action": "keep", "replace_prompt": "", "color": "gray"},
    "erase": {"mask": "", "method": "content-aware", "prompt": ""},
    "intensity": "natural",
}


def _list_photos(conn) -> None:
    rows = conn.execute(
        "SELECT id, filename, state, confidence FROM photos ORDER BY id"
    ).fetchall()
    if not rows:
        print("Queue is empty.")
        return
    for r in rows:
        conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "  - "
        print(f"  #{r['id']:<4} {r['state']:<10} conf={conf}  {r['filename']}")


def _find_tiff(row) -> Path | None:
    tiff = CONFIG.work / f"{Path(row['filename']).stem}.tiff"
    return tiff if tiff.exists() else None


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _reproduce_command(photo_id: int, analysis: dict) -> str:
    """Build a copy-paste `redo` command line that reproduces the current parameters."""
    rp = analysis.get("retouch", {})
    parts = [f"python -m pipeline.redo {photo_id}"]

    for flag, key in (
        ("--skin", "skin_smoothing"), ("--skintone", "skin_tone_correction"),
        ("--darkcircles", "reduce_dark_circles"), ("--dewlap", "reduce_dewlap"),
        ("--eyes", "brighten_eyes"), ("--iris", "iris_enhance"),
        ("--teeth", "whiten_teeth"), ("--tame-highlights", "tame_highlights"),
        ("--clothing", "clothing_contrast"),
        ("--subject-exposure", "subject_exposure"), ("--background-exposure", "background_exposure"),
        ("--hair-shimmer", "hair_shimmer"), ("--defrizz", "hair_defrizz"),
    ):
        parts.append(f"{flag} {_fmt(rp.get(key, 0.0))}")

    dp = analysis.get("develop", {})
    parts.append(f"--shadows {_fmt(dp.get('shadows', 0.0))}")
    parts.append(f"--highlights {_fmt(dp.get('highlights', 0.0))}")

    parts.append("--no-hair" if rp.get("hair_texture", 0) == 0 else f"--hair {_fmt(rp.get('hair_texture', 0.5))}")
    parts.append("--blemishes" if rp.get("remove_blemishes") else "--no-blemishes")

    if rp.get("skin_type", "auto") != "auto":
        parts.append(f"--skin-type {rp['skin_type']}")
    if abs(float(rp.get("skin_warmth", 0) or 0)) > 0.001:
        parts.append(f"--skin-warmth {_fmt(rp['skin_warmth'])}")
    if abs(float(rp.get("skin_saturation", 1.0) or 1.0) - 1.0) > 0.001:
        parts.append(f"--skin-saturation {_fmt(rp['skin_saturation'])}")
    if float(rp.get("skin_luminance", 0) or 0) > 0.001:
        parts.append(f"--skin-luminance {_fmt(rp['skin_luminance'])}")
    if float(rp.get("skin_red", 0) or 0) > 0.001:
        parts.append(f"--skin-red {_fmt(rp['skin_red'])}")
    if float(rp.get("lip_enhance", 0) or 0) > 0.001:
        parts.append(f"--lips {_fmt(rp['lip_enhance'])}")

    cl = rp.get("clothing", {}) or {}
    if cl.get("color") and cl.get("color") != "none":
        parts.append(f"--cloth-color {cl['color']}")
    for flag, key in (("--cloth-pop", "color_pop"), ("--cloth-luminance", "luminance"),
                      ("--cloth-shadows", "shadows"), ("--cloth-blacks", "blacks"),
                      ("--cloth-whites", "whites")):
        if abs(float(cl.get(key, 0) or 0)) > 0.001:
            parts.append(f"{flag} {_fmt(cl[key])}")

    if rp.get("auto_levels"):
        parts.append("--auto-levels")
        for flag, key in (("--skin-target", "auto_skin_target"),
                          ("--subject-target", "auto_subject_target"),
                          ("--background-target", "auto_background_target")):
            if key in rp:
                parts.append(f"{flag} {_fmt(rp[key])}")

    er = rp.get("erase", {}) or {}
    if er.get("mask"):
        parts.append(f"--erase-mask {er['mask']}")
        if er.get("method", "content-aware") != "content-aware":
            parts.append(f"--erase-method {er['method']}")
        if er.get("prompt"):
            parts.append(f'--erase-prompt "{er["prompt"]}"')

    bg = rp.get("background", {})
    parts.append(f"--background {bg.get('action', 'keep')}")
    if bg.get("action") == "studio":
        parts.append(f"--studio-color {bg.get('color', 'gray')}")
    if bg.get("action") == "replace" and bg.get("replace_prompt"):
        parts.append(f'--prompt "{bg["replace_prompt"]}"')

    parts.append(f"--intensity {rp.get('intensity', 'natural')}")

    wb = analysis.get("develop", {}).get("white_balance", {})
    if wb.get("mode") == "kelvin":
        parts.append(f"--wb {wb.get('temp', 6500)}")
        if wb.get("tint"):
            parts.append(f"--tint {wb['tint']}")
    else:
        parts.append("--wb camera")

    return " ".join(parts)


def _find_raw(row) -> Path | None:
    for candidate in (Path(row["path"]), CONFIG.archive / row["filename"]):
        if candidate.exists():
            return candidate
    return None


def _reanalyze(row) -> dict:
    """Re-run the AI vision analysis for a photo and return the fresh analysis dict."""
    import anthropic
    from .analysis import analyze_preview
    from .preview import make_preview

    preview = Path(row["preview_path"]) if row["preview_path"] else None
    if preview is None or not preview.exists():
        raw = _find_raw(row)
        if raw is None:
            raise RuntimeError(f"No preview or RAW found for {row['filename']} to analyze")
        preview = make_preview(raw)
    log.info("Re-running AI analysis on %s (model=%s)", row["filename"], CONFIG.model)
    result = analyze_preview(preview, anthropic.Anthropic())
    log.info("New analysis: portrait=%s confidence=%.2f", result.retouch.is_portrait, result.confidence)
    return json.loads(result.model_dump_json())


def _parse_id_spec(spec: str) -> list[int]:
    """Parse '91', '91-98', '91,93,95', or a combo like '91-93,97' into a list of ids."""
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):   # a range (not just a leading minus)
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            ids.extend(range(min(lo, hi), max(lo, hi) + 1))
        else:
            ids.append(int(part))
    seen: set[int] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def _show_photo(conn, photo_id: int) -> None:
    row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if row is None:
        print(f"#{photo_id} not in the queue")
        return
    if not row["analysis_json"]:
        print(f"#{photo_id} {row['filename']} (state={row['state']}) — no analysis yet")
        return
    analysis = json.loads(row["analysis_json"])
    analysis["retouch"] = {**_DEFAULT_RETOUCH, **analysis.get("retouch", {})}
    conf = f"{row['confidence']:.2f}" if row["confidence"] is not None else "n/a"
    print(f"#{row['id']} {row['filename']} (state={row['state']}, confidence={conf})")
    if analysis.get("scene_description"):
        print(f"scene: {analysis['scene_description']}")
    print(json.dumps({k: analysis[k] for k in ("develop", "retouch") if k in analysis}, indent=2))
    print("\n# reproduce this render (retouch + white balance):")
    print(_reproduce_command(row["id"], analysis))


def _dump_mask(conn, photo_id: int) -> None:
    """Save a [original | skin-selection] overlay so the skin mask can be inspected."""
    from .retouch import dump_skin_mask
    row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if row is None:
        print(f"#{photo_id} not in the queue")
        return
    tiff = _find_tiff(row)
    if tiff is None:
        raw = _find_raw(row)
        if raw is None:
            print(f"#{photo_id}: no developed TIFF or RAW found — cannot build the mask")
            return
        analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else {"develop": {}}
        CONFIG.work.mkdir(parents=True, exist_ok=True)
        tiff = develop(raw, json.dumps(analysis), CONFIG.work)
    out = CONFIG.output / f"_mask_{Path(row['filename']).stem}_{photo_id}.jpg"
    path, cov = dump_skin_mask(tiff, out)
    print(f"#{photo_id} -> {path}  (skin selection = {100 * cov:.0f}% of frame)")


def _process_photo(conn, photo_id: int, args, overrides: dict | None) -> Path:
    """Render one photo, applying overrides. Raises on any problem so a batch can continue."""
    row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"#{photo_id} not in the queue")

    if args.reanalyze:
        analysis = _reanalyze(row)
    else:
        if not row["analysis_json"]:
            raise RuntimeError(f"#{photo_id} has no analysis yet (state={row['state']}) — "
                               "use --reanalyze")
        analysis = json.loads(row["analysis_json"])
    analysis["retouch"] = {**_DEFAULT_RETOUCH, **analysis.get("retouch", {})}

    apply_overrides(analysis, overrides)
    if not background_overridden(overrides):
        force_keep_background(analysis)
    develop_changed = develop_overridden(overrides)
    analysis_json = json.dumps(analysis)

    tiff = None if (args.from_raw or develop_changed or args.reanalyze) else _find_tiff(row)
    if tiff is None:
        raw = _find_raw(row)
        if raw is None:
            raise RuntimeError(f"neither a developed TIFF nor the RAW for {row['filename']} was found")
        log.info("Developing %s from RAW", raw.name)
        CONFIG.work.mkdir(parents=True, exist_ok=True)
        tiff = develop(raw, analysis_json, CONFIG.work)

    result = retouch(tiff, analysis_json)
    if result == tiff:
        log.warning("#%d nothing to retouch with these settings — output is the developed TIFF", photo_id)

    out_path = publish_done(row["id"], row["filename"], result)
    write_rapidraw_sidecar(row["filename"], analysis_json)
    cleanup_intermediates(Path(row["filename"]).stem)
    with conn:
        conn.execute("UPDATE photos SET analysis_json = ?, confidence = ? WHERE id = ?",
                     (analysis_json, analysis.get("confidence"), row["id"]))
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-8s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photos", nargs="?", help="queue id, range or list: 91, 91-98, 91,93,95")
    p.add_argument("--list", action="store_true", help="list photos in the queue")
    p.add_argument("--show", action="store_true", help="print current params (per photo) and exit")
    p.add_argument("--dump-mask", action="store_true",
                   help="save a [original | skin-selection] overlay JPEG to inspect the mask, then exit")
    add_override_args(p)
    p.add_argument("--reanalyze", action="store_true",
                   help="re-run the AI vision analysis before rendering")
    p.add_argument("--from-raw", action="store_true",
                   help="re-develop the RAW before retouching (instead of reusing the TIFF)")
    args = p.parse_args()

    if args.subject_only:
        CONFIG.analyze_subject_only = True
    if args.skin_exposure:
        CONFIG.analyze_skin_exposure = True

    conn = db.connect(CONFIG.db_path)

    if args.list or args.photos is None:
        _list_photos(conn)
        return

    try:
        ids = _parse_id_spec(args.photos)
    except ValueError:
        sys.exit(f"Invalid id spec {args.photos!r} — use e.g. 91, 91-98, or 91,93,95")
    if not ids:
        sys.exit("No photo ids given")

    if args.show:
        for pid in ids:
            _show_photo(conn, pid)
            if len(ids) > 1:
                print()
        return

    if args.dump_mask:
        for pid in ids:
            _dump_mask(conn, pid)
        return

    overrides = overrides_from_args(args)
    total, done = len(ids), 0
    for n, pid in enumerate(ids, 1):
        if total > 1:
            log.info("[%d/%d] photo #%d", n, total, pid)
        try:
            out_path = _process_photo(conn, pid, args, overrides)
            done += 1
            print(f"#{pid} -> {out_path}")
        except Exception as exc:
            log.error("#%d skipped: %s", pid, exc)

    print(f"\nDone: {done}/{total} photo(s)")


if __name__ == "__main__":
    main()
