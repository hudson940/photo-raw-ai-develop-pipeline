"""Work queue, backed by SQLite (default) or PostgreSQL.

States: pending -> previewed -> analyzed -> developed -> retouched -> done
Side states: review (needs human), failed (exhausted retries).
Stages 4+ (developed/retouched/done) are transitioned by later pipeline stages.

The backend is chosen by CONFIG.database_url: empty -> SQLite file at db_path;
a `postgres://` / `postgresql://` URL -> PostgreSQL (psycopg 3). A thin wrapper
(`Conn` / `Cur`) presents one API over both drivers so the rest of the codebase
keeps writing plain `conn.execute("... WHERE id = ?", (id,))` with `?` params and
dict-style row access, regardless of which database is in use.
"""

import re
import sqlite3
import threading
import time
from pathlib import Path

from .config import CONFIG

# --------------------------------------------------------------------------- SQL dialect

# Raised on a UNIQUE/PK violation, whichever backend is active. psycopg's
# IntegrityError is only importable when psycopg is installed, so build the
# catch-tuple lazily and fall back to just sqlite3's when it is not.
try:  # pragma: no cover - depends on optional dependency
    import psycopg
    IntegrityError: tuple = (sqlite3.IntegrityError, psycopg.errors.IntegrityError)
except Exception:  # psycopg not installed -> SQLite-only deployment
    psycopg = None
    IntegrityError = (sqlite3.IntegrityError,)


def _is_postgres() -> bool:
    return CONFIG.database_url.startswith(("postgres://", "postgresql://"))


def _translate_ddl(sql: str) -> str:
    """Rewrite the small set of SQLite-isms in our schema DDL to PostgreSQL."""
    sql = re.sub(r"INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY", sql,
                 flags=re.IGNORECASE)
    sql = re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql, flags=re.IGNORECASE)
    return sql


def _translate_dml(sql: str) -> str:
    """Rewrite a DML statement for PostgreSQL: `?` placeholders -> `%s`, and
    SQLite's `INSERT OR IGNORE` -> `INSERT ... ON CONFLICT DO NOTHING`."""
    ignore = re.search(r"INSERT\s+OR\s+IGNORE", sql, flags=re.IGNORECASE)
    if ignore:
        sql = re.sub(r"INSERT\s+OR\s+IGNORE", "INSERT", sql, flags=re.IGNORECASE)
    sql = sql.replace("?", "%s")
    if ignore:
        sql += " ON CONFLICT DO NOTHING"
    return sql


# --------------------------------------------------------------------------- wrapper

class Cur:
    """Cursor wrapper: uniform fetchone/fetchall/iteration/rowcount over both drivers."""

    def __init__(self, raw):
        self._raw = raw

    def fetchone(self):
        return self._raw.fetchone()

    def fetchall(self):
        return self._raw.fetchall()

    def __iter__(self):
        return iter(self._raw)

    @property
    def rowcount(self) -> int:
        return self._raw.rowcount


class Conn:
    """Connection wrapper presenting the SQLite-style API used across the codebase.

    A per-connection lock serializes access so a single Conn can be shared between
    threads (the SQLite path relied on this; PostgreSQL connections are likewise not
    safe for concurrent use). Transactions use `with conn:` — commit on success,
    rollback on exception — without closing the connection.
    """

    def __init__(self, raw, kind: str):
        self.raw = raw
        self.kind = kind                # "sqlite" | "postgres"
        self._lock = threading.RLock()
        self._tx = None                 # active psycopg transaction context, if any

    def _sql(self, sql: str) -> str:
        return _translate_dml(sql) if self.kind == "postgres" else sql

    def execute(self, sql: str, params: tuple = ()) -> Cur:
        with self._lock:
            if self.kind == "postgres":
                cur = self.raw.cursor()
                cur.execute(self._sql(sql), params)
                return Cur(cur)
            return Cur(self.raw.execute(sql, params))

    def executemany(self, sql: str, seq) -> Cur:
        with self._lock:
            if self.kind == "postgres":
                cur = self.raw.cursor()
                cur.executemany(self._sql(sql), list(seq))
                return Cur(cur)
            return Cur(self.raw.executemany(sql, seq))

    def executescript(self, script: str) -> None:
        with self._lock:
            if self.kind == "postgres":
                for stmt in _translate_ddl(script).split(";"):
                    if stmt.strip():
                        self.raw.execute(stmt)
            else:
                self.raw.executescript(script)

    def insert(self, sql: str, params: tuple = ()) -> int:
        """Run a single-row INSERT and return the new integer primary key (lastrowid)."""
        with self._lock:
            if self.kind == "postgres":
                cur = self.raw.cursor()
                cur.execute(self._sql(sql) + " RETURNING id", params)
                return cur.fetchone()["id"]
            return self.raw.execute(sql, params).lastrowid

    def columns(self, table: str) -> set:
        """Column names of a table — replaces SQLite's `PRAGMA table_info`."""
        if self.kind == "postgres":
            rows = self.execute(
                "SELECT column_name AS name FROM information_schema.columns"
                " WHERE table_name = ?", (table,)).fetchall()
            return {r["name"] for r in rows}
        return {r["name"] for r in self.raw.execute(f"PRAGMA table_info({table})")}

    def commit(self) -> None:
        self.raw.commit()

    def close(self) -> None:
        self.raw.close()

    def __enter__(self):
        self._lock.acquire()
        if self.kind == "postgres":
            self._tx = self.raw.transaction()
            self._tx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.kind == "postgres":
                self._tx.__exit__(exc_type, exc, tb)
                self._tx = None
            elif exc_type is None:
                self.raw.commit()
            else:
                self.raw.rollback()
        finally:
            self._lock.release()
        return False


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
    selected INTEGER,           -- operator pick: NULL undecided / 1 selected / 0 not-selected
    deliver TEXT                -- which version to deliver/download: NULL/'developed' or 'original'
);
CREATE INDEX IF NOT EXISTS idx_photos_state ON photos(state);
"""

# Columns added after the first release; ALTER-ed in on connect for existing DBs.
_MIGRATIONS = {
    "owner": "ALTER TABLE photos ADD COLUMN owner TEXT",
    "quality_json": "ALTER TABLE photos ADD COLUMN quality_json TEXT",
    "selected": "ALTER TABLE photos ADD COLUMN selected INTEGER",
    "deliver": "ALTER TABLE photos ADD COLUMN deliver TEXT",
    "raw_removed": "ALTER TABLE photos ADD COLUMN raw_removed INTEGER",
}


def _migrate(conn: Conn) -> None:
    have = conn.columns("photos")
    with conn:
        for col, ddl in _MIGRATIONS.items():
            if col not in have:
                conn.execute(ddl)
        # index on owner only after the column is guaranteed to exist
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_owner ON photos(owner)")


def connect(db_path: Path) -> Conn:
    """Open the work-queue database. Uses PostgreSQL when CONFIG.database_url is a
    postgres URL, otherwise a SQLite file at `db_path`."""
    if _is_postgres():
        if psycopg is None:
            raise RuntimeError(
                "DATABASE_URL points at PostgreSQL but psycopg is not installed "
                "(pip install 'psycopg[binary]')")
        raw = psycopg.connect(CONFIG.database_url, autocommit=True,
                              row_factory=psycopg.rows.dict_row)
        conn = Conn(raw, "postgres")
    else:
        raw = sqlite3.connect(db_path, check_same_thread=False)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA journal_mode=WAL")
        conn = Conn(raw, "sqlite")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def enqueue(conn: Conn, path: Path, owner: str | None = None) -> bool:
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
    except IntegrityError:
        return False


def claim_next_pending(conn: Conn):
    """Fetch the oldest pending row whose retry backoff has elapsed."""
    return conn.execute(
        "SELECT * FROM photos WHERE state = 'pending' AND next_attempt_at <= ?"
        " ORDER BY id LIMIT 1",
        (time.time(),),
    ).fetchone()


def set_state(conn: Conn, photo_id: int, state: str, **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    sql = f"UPDATE photos SET state = ?, updated_at = ?{', ' + cols if cols else ''} WHERE id = ?"
    with conn:
        conn.execute(sql, (state, time.time(), *fields.values(), photo_id))


def record_failure(
    conn: Conn,
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


def requeue(conn: Conn, states: tuple[str, ...] = ("previewed",),
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


def counts_by_state(conn: Conn) -> dict[str, int]:
    rows = conn.execute("SELECT state, COUNT(*) AS n FROM photos GROUP BY state").fetchall()
    return {r["state"]: r["n"] for r in rows}
