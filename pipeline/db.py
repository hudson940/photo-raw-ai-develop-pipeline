"""SQLite-backed work queue.

States: pending -> previewed -> analyzed -> developed -> retouched -> done
Side states: review (needs human), failed (exhausted retries).
Stages 4+ (developed/retouched/done) are transitioned by later pipeline stages.
"""

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    preview_path TEXT,
    analysis_json TEXT,
    confidence REAL,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    owner TEXT,                 -- username of the editor who owns this photo (NULL = shared/admin)
    quality_json TEXT,          -- pre-processing blur/exposure assessment (see pipeline.quality)
    selected INTEGER            -- operator pick: NULL undecided / 1 selected / 0 not-selected
);
CREATE INDEX IF NOT EXISTS idx_photos_state ON photos(state);
"""

# Columns added after the first release; ALTER-ed in on connect for existing DBs.
_MIGRATIONS = {
    "owner": "ALTER TABLE photos ADD COLUMN owner TEXT",
    "quality_json": "ALTER TABLE photos ADD COLUMN quality_json TEXT",
    "selected": "ALTER TABLE photos ADD COLUMN selected INTEGER",
}


def _migrate(conn: sqlite3.Connection) -> None:
    have = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
    with conn:
        for col, ddl in _MIGRATIONS.items():
            if col not in have:
                conn.execute(ddl)
        # index on owner only after the column is guaranteed to exist
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_owner ON photos(owner)")


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def enqueue(conn: sqlite3.Connection, path: Path, owner: str | None = None) -> bool:
    """Insert a new photo in 'pending' state. Returns False if already known.

    `owner` is the editor username the photo belongs to (from its inbox subfolder);
    NULL means a shared/admin photo visible to every operator.
    """
    now = time.time()
    try:
        with conn:
            conn.execute(
                "INSERT INTO photos (path, filename, state, owner, created_at, updated_at)"
                " VALUES (?, ?, 'pending', ?, ?, ?)",
                (str(path), path.name, owner, now, now),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def claim_next_pending(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Fetch the oldest pending row whose retry backoff has elapsed."""
    return conn.execute(
        "SELECT * FROM photos WHERE state = 'pending' AND next_attempt_at <= ?"
        " ORDER BY id LIMIT 1",
        (time.time(),),
    ).fetchone()


def set_state(conn: sqlite3.Connection, photo_id: int, state: str, **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    sql = f"UPDATE photos SET state = ?, updated_at = ?{', ' + cols if cols else ''} WHERE id = ?"
    with conn:
        conn.execute(sql, (state, time.time(), *fields.values(), photo_id))


def record_failure(
    conn: sqlite3.Connection,
    photo_id: int,
    error: str,
    max_attempts: int,
    backoff_base_s: float,
) -> str:
    """Bump the attempt counter; requeue with backoff or mark failed. Returns the new state."""
    row = conn.execute("SELECT attempts FROM photos WHERE id = ?", (photo_id,)).fetchone()
    attempts = row["attempts"] + 1
    if attempts >= max_attempts:
        state, next_at = "failed", 0
    else:
        state, next_at = "pending", time.time() + backoff_base_s * (2 ** (attempts - 1))
    with conn:
        conn.execute(
            "UPDATE photos SET state = ?, attempts = ?, next_attempt_at = ?,"
            " error = ?, updated_at = ? WHERE id = ?",
            (state, attempts, next_at, error, time.time(), photo_id),
        )
    return state


def requeue(conn: sqlite3.Connection, states: tuple[str, ...] = ("previewed",),
            clear_analysis: bool = False) -> int:
    """Reset photos in the given states back to 'pending' (ready now) so the worker
    reprocesses them. Defaults to 'previewed' rows left stuck by an interrupted run.

    With clear_analysis=True, also wipe the cached analysis so the worker re-runs the
    AI vision analysis from scratch instead of reusing the stored parameters.
    """
    placeholders = ",".join("?" * len(states))
    extra = ", analysis_json=NULL, confidence=NULL" if clear_analysis else ""
    with conn:
        cur = conn.execute(
            f"UPDATE photos SET state='pending', next_attempt_at=0, updated_at=?{extra} "
            f"WHERE state IN ({placeholders})",
            (time.time(), *states),
        )
    return cur.rowcount


def counts_by_state(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT state, COUNT(*) AS n FROM photos GROUP BY state").fetchall()
    return {r["state"]: r["n"] for r in rows}
