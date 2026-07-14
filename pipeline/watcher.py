"""Inbox watcher: detects new RAW files and enqueues them once fully written.

Uses watchdog's PollingObserver — inotify does not work reliably on Windows
mounts (/mnt/c) or Samba shares, which is where the inbox will live.
"""

import logging
import sqlite3
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from .config import CONFIG, RAW_EXTENSIONS
from . import db

log = logging.getLogger("watcher")


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def owner_for(path: Path) -> str | None:
    """Editor username from a photo's inbox subfolder: inbox/<user>/f.CR3 -> '<user>'.
    A file dropped directly in the inbox root has no owner (shared/admin)."""
    try:
        rel = path.resolve().relative_to(CONFIG.inbox.resolve())
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def wait_until_stable(path: Path, delay_s: float, checks: int = 2) -> bool:
    """Camera/network transfers are slow; wait until the size stops changing."""
    try:
        last = path.stat().st_size
    except OSError:
        return False
    for _ in range(checks * 30):  # give up after ~a minute per check cycle
        time.sleep(delay_s)
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == last and size > 0:
            checks -= 1
            if checks == 0:
                return True
        else:
            last = size
    return False


class RawFileHandler(FileSystemEventHandler):
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def on_created(self, event):
        self._handle(event)

    def on_moved(self, event):
        # renames into the inbox (e.g. atomic copy tools) land here
        event.src_path = event.dest_path
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if not is_raw(path):
            return
        if not wait_until_stable(path, CONFIG.stable_check_delay_s):
            log.warning("File never stabilized, skipping: %s", path)
            return
        owner = owner_for(path)
        if db.enqueue(self.conn, path, owner):
            log.info("Enqueued %s%s", path.name, f" (owner={owner})" if owner else "")


def scan_existing(conn: sqlite3.Connection) -> int:
    """Pick up files already sitting in the inbox at startup (recursively, so per-user
    subfolders inbox/<user>/ are ingested and attributed to that user)."""
    added = 0
    for path in sorted(CONFIG.inbox.rglob("*")):
        if path.is_file() and is_raw(path):
            if db.enqueue(conn, path, owner_for(path)):
                log.info("Enqueued existing file %s", path.name)
                added += 1
    return added


def start_observer(conn: sqlite3.Connection) -> PollingObserver:
    observer = PollingObserver(timeout=CONFIG.scan_interval_s)
    # recursive so per-user subfolders (inbox/<user>/) are watched too
    observer.schedule(RawFileHandler(conn), str(CONFIG.inbox), recursive=True)
    observer.start()
    log.info("Watching %s (poll every %.0fs)", CONFIG.inbox, CONFIG.scan_interval_s)
    return observer
