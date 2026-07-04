"""Re-run retouch on a processed photo with your own choices.

The AI proposes retouch parameters in Stage 3; this CLI lets you override any
of them per photo and re-render without re-running analysis or development:

    python -m pipeline.redo --list
    python -m pipeline.redo 3 --show
    python -m pipeline.redo 3 --skin 0.3 --background blur
    python -m pipeline.redo 3 --background replace --prompt "soft window light, bright interior"
    python -m pipeline.redo 3 --clothing 0.4 --eyes 0.2 --intensity subtle

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
from .output import cleanup_intermediates, publish_done
from .retouch import retouch

log = logging.getLogger("redo")

_DEFAULT_RETOUCH = {
    "is_portrait": True,
    "skin_smoothing": 0.0,
    "remove_blemishes": False,
    "skin_tone_correction": 0.0,
    "reduce_dark_circles": 0.0,
    "reduce_dewlap": 0.0,
    "brighten_eyes": 0.0,
    "iris_enhance": 0.0,
    "whiten_teeth": 0.0,
    "clothing_contrast": 0.0,
    "background": {"action": "keep", "replace_prompt": ""},
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


def _find_raw(row) -> Path | None:
    for candidate in (Path(row["path"]), CONFIG.archive / row["filename"]):
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-8s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photo_id", nargs="?", type=int, help="queue id (see --list)")
    p.add_argument("--list", action="store_true", help="list photos in the queue")
    p.add_argument("--show", action="store_true", help="print current retouch params and exit")
    p.add_argument("--skin", type=float, metavar="0..1", help="skin smoothing strength")
    p.add_argument("--blemishes", dest="blemishes", action="store_true", default=None,
                   help="erase pimples, acne marks, skin tags, stray hairs")
    p.add_argument("--no-blemishes", dest="blemishes", action="store_false",
                   help="disable blemish removal")
    p.add_argument("--skintone", type=float, metavar="0..1",
                   help="even out skin tone / healthy color correction")
    p.add_argument("--darkcircles", type=float, metavar="0..1",
                   help="reduce under-eye dark circles / eye bags")
    p.add_argument("--dewlap", type=float, metavar="0..1",
                   help="reduce a dewlap / double chin (subtle under-chin lift)")
    p.add_argument("--eyes", type=float, metavar="0..1", help="eye-white (sclera) brightening")
    p.add_argument("--iris", type=float, metavar="0..1", help="iris saturation/clarity pop")
    p.add_argument("--teeth", type=float, metavar="0..1", help="teeth whitening strength")
    p.add_argument("--clothing", type=float, metavar="0..1", help="clothing contrast strength")
    p.add_argument("--background", choices=["keep", "blur", "smooth", "studio", "replace"],
                   help="background handling (smooth = de-wrinkle a backdrop)")
    p.add_argument("--prompt", help="background description for --background replace")
    p.add_argument("--wb", metavar="camera|KELVIN",
                   help="white balance: 'camera' (accurate as-shot) or a Kelvin number "
                        "like 5500 (higher = warmer). Re-develops from RAW.")
    p.add_argument("--tint", type=int, metavar="-50..50",
                   help="green-magenta tint for --wb kelvin mode (-50 green .. +50 magenta)")
    p.add_argument("--intensity", choices=["subtle", "natural", "polished"],
                   help="overall retouch strength")
    p.add_argument("--from-raw", action="store_true",
                   help="re-develop the RAW before retouching (instead of reusing the TIFF)")
    args = p.parse_args()

    conn = db.connect(CONFIG.db_path)

    if args.list or args.photo_id is None:
        _list_photos(conn)
        return

    row = conn.execute("SELECT * FROM photos WHERE id = ?", (args.photo_id,)).fetchone()
    if row is None:
        sys.exit(f"No photo #{args.photo_id} in the queue (try --list)")
    if not row["analysis_json"]:
        sys.exit(f"Photo #{args.photo_id} has no analysis yet (state={row['state']}) — "
                 "it must pass Stage 3 first")

    analysis = json.loads(row["analysis_json"])
    rp = {**_DEFAULT_RETOUCH, **analysis.get("retouch", {})}

    if args.show:
        print(f"#{row['id']} {row['filename']} (state={row['state']})")
        print(json.dumps(rp, indent=2))
        return

    # apply overrides
    if args.skin is not None:
        rp["skin_smoothing"], rp["is_portrait"] = args.skin, True
    if args.blemishes is not None:
        rp["remove_blemishes"], rp["is_portrait"] = args.blemishes, True
    if args.skintone is not None:
        rp["skin_tone_correction"], rp["is_portrait"] = args.skintone, True
    if args.darkcircles is not None:
        rp["reduce_dark_circles"], rp["is_portrait"] = args.darkcircles, True
    if args.dewlap is not None:
        rp["reduce_dewlap"], rp["is_portrait"] = args.dewlap, True
    if args.eyes is not None:
        rp["brighten_eyes"], rp["is_portrait"] = args.eyes, True
    if args.iris is not None:
        rp["iris_enhance"], rp["is_portrait"] = args.iris, True
    if args.teeth is not None:
        rp["whiten_teeth"], rp["is_portrait"] = args.teeth, True
    if args.clothing is not None:
        rp["clothing_contrast"] = args.clothing
    if args.background is not None:
        rp.setdefault("background", {})
        rp["background"]["action"] = args.background
    if args.prompt is not None:
        rp.setdefault("background", {})
        rp["background"]["replace_prompt"] = args.prompt
    if args.intensity is not None:
        rp["intensity"] = args.intensity

    analysis["retouch"] = rp

    # white balance lives in the develop block; changing it needs a re-develop
    wb_changed = False
    if args.wb is not None or args.tint is not None:
        dp = analysis.setdefault("develop", {})
        wb = dp.setdefault("white_balance", {})
        if args.wb is not None:
            if args.wb.lower() == "camera":
                wb["mode"] = "camera"
            else:
                try:
                    wb["mode"], wb["temp"] = "kelvin", int(args.wb)
                except ValueError:
                    sys.exit(f"--wb must be 'camera' or a Kelvin number, got {args.wb!r}")
        if args.tint is not None:
            wb["mode"], wb["tint"] = "kelvin", args.tint
        wb_changed = True

    analysis_json = json.dumps(analysis)

    tiff = None if (args.from_raw or wb_changed) else _find_tiff(row)
    if tiff is None:
        raw = _find_raw(row)
        if raw is None:
            sys.exit(f"Neither a developed TIFF nor the RAW for {row['filename']} was found")
        log.info("Developing %s from RAW", raw.name)
        CONFIG.work.mkdir(parents=True, exist_ok=True)
        tiff = develop(raw, analysis_json, CONFIG.work)

    result = retouch(tiff, analysis_json)
    if result == tiff:
        log.warning("Nothing to retouch with these settings — output is the developed TIFF")

    out_path = publish_done(row["id"], row["filename"], result)
    cleanup_intermediates(Path(row["filename"]).stem)

    with conn:
        conn.execute("UPDATE photos SET analysis_json = ? WHERE id = ?",
                     (analysis_json, row["id"]))

    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    main()
