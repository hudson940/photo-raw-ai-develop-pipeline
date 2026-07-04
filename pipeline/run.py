"""Entry point: python -m pipeline.run

Starts the inbox watcher (background thread) and the worker loop (main thread).
Ctrl-C to stop.
"""

import logging
import os
import sys
from pathlib import Path

from .config import CONFIG
from . import db, watcher, worker
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-8s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    CONFIG.ensure_dirs()
    check_tools()
    check_credentials()

    conn = db.connect(CONFIG.db_path)
    added = watcher.scan_existing(conn)
    if added:
        logging.info("Startup scan enqueued %d file(s)", added)

    observer = watcher.start_observer(conn)
    try:
        worker.run_forever(conn)
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        observer.stop()
        observer.join()
        conn.close()


if __name__ == "__main__":
    main()
