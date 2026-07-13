"""Entry point: python -m pipeline.run

Starts the inbox watcher (background thread) and the worker loop (main thread).
Ctrl-C to stop.

Retouch/white-balance flags (same as `pipeline.redo`) can be passed to force
settings onto every photo the worker processes, e.g.:

    python -m pipeline.run --skin 0.3 --dewlap 0.3 --wb camera --background blur
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import CONFIG
from . import db, watcher, worker
from .overrides import add_override_args, describe, overrides_from_args
from .preview import find_tool


def check_credentials() -> None:
    """Fail fast if no Anthropic credential source exists — the worker can't run without one."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return
    if (Path.home() / ".config" / "anthropic" / "credentials").exists():
        return  # `ant auth login` profile; the SDK picks it up automatically
    logging.error("No Anthropic credentials found. Set one of:")
    logging.error("  export ANTHROPIC_API_KEY=sk-ant-...   (from console.anthropic.com)")
    logging.error("  or put ANTHROPIC_API_KEY=sk-ant-... in %s/.env (chmod 600)", Path(__file__).resolve().parent.parent)
    logging.error("  or run `ant auth login` if you use the Anthropic CLI")
    sys.exit(1)


def check_tools() -> None:
    have_exiftool = find_tool("exiftool") is not None
    try:
        import rawpy  # noqa: F401
        have_rawpy = True
    except ImportError:
        have_rawpy = False
    have_darktable = find_tool("darktable-cli") is not None

    if not have_exiftool:
        logging.warning("exiftool not found — previews will be decoded from RAW (slower)")
    if not (have_rawpy or have_darktable):
        logging.warning("neither rawpy nor darktable-cli available — RAWs without embedded previews will fail")
    if not any((have_exiftool, have_rawpy, have_darktable)):
        logging.error("No RAW preview tool available; Stage 2 cannot run.")
        logging.error("Fix with: pip install rawpy   (or: sudo apt install exiftool darktable)")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--once", action="store_true",
                   help="process all currently-pending photos, then exit (no watching)")
    p.add_argument("--requeue-stuck", action="store_true",
                   help="reset photos stuck in 'previewed' (interrupted runs) back to pending first")
    p.add_argument("--requeue-review", action="store_true",
                   help="also requeue photos parked in 'review' (low confidence)")
    p.add_argument("--reanalyze", action="store_true",
                   help="with --requeue-*, wipe cached analysis so the AI re-analyzes from scratch")
    add_override_args(p)
    args = p.parse_args()
    overrides = overrides_from_args(args)
    if args.subject_only:
        CONFIG.analyze_subject_only = True
    if args.skin_exposure:
        CONFIG.analyze_skin_exposure = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-8s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    CONFIG.ensure_dirs()
    check_tools()
    check_credentials()

    if overrides:
        logging.info("Forcing overrides on every photo: %s", describe(overrides))

    conn = db.connect(CONFIG.db_path)

    if args.requeue_stuck or args.requeue_review:
        states = ("previewed",) + (("review",) if args.requeue_review else ())
        n = db.requeue(conn, states, clear_analysis=args.reanalyze)
        logging.info("Requeued %d stuck photo(s) from %s%s", n, "/".join(states),
                     " (fresh AI analysis)" if args.reanalyze else "")

    added = watcher.scan_existing(conn)
    if added:
        logging.info("Startup scan enqueued %d file(s)", added)

    # One-shot: drain the queue and exit (no watcher, no idle loop).
    if args.once:
        try:
            worker.drain(conn, overrides)
        except KeyboardInterrupt:
            logging.info("Interrupted")
        finally:
            conn.close()
        return

    observer = watcher.start_observer(conn)
    try:
        worker.run_forever(conn, overrides)
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        observer.stop()
        observer.join()
        conn.close()


if __name__ == "__main__":
    main()
