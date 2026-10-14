"""Event catalog: one small row per event, request, snapshot and model bundle.

The catalog never stores arrays; the event files are the ground truth. It has
exactly one writer, enforced by an exclusive lock file, so no process ever
waits on, or polls, a SQLite lock. Producers (campaign jobs, teachers, coupled
runs) only drop event files; the single control-plane process ingests them.
Readers (status, trainers, notebooks) open the catalog read-only.

Keep the catalog on a local filesystem: SQLite locking and WAL shared memory
are not reliable on NFS/Lustre.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,  -- monotonic ingest sequence
    sha256          TEXT NOT NULL UNIQUE,
    path            TEXT NOT NULL,
    size            INTEGER NOT NULL,
    case_id         TEXT NOT NULL,
    b2_call_index   INTEGER NOT NULL,
    event_kind      TEXT NOT NULL,
    repeat_index    INTEGER NOT NULL,
    repeat_count    INTEGER NOT NULL,
    used_by_b2      INTEGER,                            -- NULL when the event does not record it
    background_hash TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    created_local   TEXT,
    solps_iter_git  TEXT,
    eirene_git      TEXT,
    b2_5_git        TEXT,
    origin          TEXT NOT NULL,
    request_id      TEXT REFERENCES requests(request_id),
    ingested_utc    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_background ON events(background_hash);
CREATE INDEX IF NOT EXISTS events_case ON events(case_id);

CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    controls    TEXT NOT NULL,              -- JSON: sampled run controls (source_params.json)
    source      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejected (
    path        TEXT PRIMARY KEY,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    reason      TEXT NOT NULL,
    seen_utc    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    request_id      TEXT PRIMARY KEY,
    background_hash TEXT NOT NULL UNIQUE,
    case_id         TEXT NOT NULL,
    candidate_ref   TEXT NOT NULL,
    bundle_id       TEXT,
    score           TEXT NOT NULL,          -- JSON: errbar ratio, novelty, rank
    status          TEXT NOT NULL CHECK (status IN ('open', 'fulfilled', 'failed')),
    created_utc     TEXT NOT NULL,
    closed_utc      TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id     TEXT PRIMARY KEY,       -- content hash of the manifest
    path            TEXT NOT NULL,
    high_water      INTEGER NOT NULL,       -- max event_id eligible for this snapshot
    n_events        INTEGER NOT NULL,
    created_utc     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bundles (
    bundle_id       TEXT PRIMARY KEY,
    snapshot_id     TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    learner         TEXT NOT NULL,
    path            TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('candidate', 'promoted', 'retired', 'rejected')),
    metrics         TEXT NOT NULL,          -- JSON
    created_utc     TEXT NOT NULL,
    promoted_utc    TEXT
);

CREATE TABLE IF NOT EXISTS audit (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    utc         TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL               -- JSON
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class WriterBusy(RuntimeError):
    """Another process already owns the catalog writer lock."""


class Catalog:
    """Read-only by default; `Catalog.writer(path)` takes the exclusive writer lock."""

    def __init__(self, path: str | Path, conn: sqlite3.Connection, lock_fd: int | None = None):
        self.path = Path(path)
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self._lock_fd = lock_fd

    @classmethod
    def writer(cls, path: str | Path) -> "Catalog":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(f"{path}.writer.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise WriterBusy(f"{path} is owned by another writer") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        conn = sqlite3.connect(path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        if "used_by_b2" not in {row[1] for row in conn.execute("PRAGMA table_info(events)")}:
            conn.execute("ALTER TABLE events ADD COLUMN used_by_b2 INTEGER")
        return cls(path, conn, fd)

    @classmethod
    def reader(cls, path: str | Path) -> "Catalog":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return cls(path, conn)

    @property
    def writable(self) -> bool:
        return self._lock_fd is not None

    def close(self) -> None:
        self.conn.close()
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ writes

    def _require_writer(self) -> None:
        if not self.writable:
            raise PermissionError("catalog opened read-only")

    def transaction(self):
        self._require_writer()
        return _Transaction(self.conn)

    def insert(self, table: str, row: dict) -> int:
        self._require_writer()
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        cur = self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
        return cur.lastrowid

    def audit(self, action: str, **detail) -> None:
        self.insert("audit", {"utc": utcnow(), "action": action,
                              "detail": json.dumps(detail, sort_keys=True, default=str)})

    # ------------------------------------------------------------------- reads

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql: str, params: tuple = ()):
        row = self.conn.execute(sql, params).fetchone()
        return None if row is None else row[0]

    def high_water(self) -> int:
        return int(self.scalar("SELECT COALESCE(MAX(event_id), 0) FROM events"))

    def known_paths(self) -> dict[str, tuple[int, float | None]]:
        """path -> (size, mtime) for ingested and rejected files."""
        known = {r["path"]: (r["size"], None) for r in self.query("SELECT path, size FROM events")}
        known.update({r["path"]: (r["size"], r["mtime"])
                      for r in self.query("SELECT path, size, mtime FROM rejected")})
        return known

    def promoted_bundle(self) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM bundles WHERE status='promoted' ORDER BY promoted_utc DESC LIMIT 1")
        return rows[0] if rows else None

    def latest_snapshot(self) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM snapshots ORDER BY high_water DESC, created_utc DESC LIMIT 1")
        return rows[0] if rows else None

    def summary(self) -> dict:
        def counts(sql):
            return {r[0]: r[1] for r in self.query(sql)}
        return {
            "events": self.scalar("SELECT COUNT(*) FROM events"),
            "cases": self.scalar("SELECT COUNT(DISTINCT case_id) FROM events"),
            "cases_with_controls": self.scalar("SELECT COUNT(*) FROM cases"),
            "backgrounds": self.scalar("SELECT COUNT(DISTINCT background_hash) FROM events"),
            "events_by_origin": counts("SELECT origin, COUNT(*) FROM events GROUP BY origin"),
            "events_by_kind": counts("SELECT event_kind, COUNT(*) FROM events GROUP BY event_kind"),
            "rejected": self.scalar("SELECT COUNT(*) FROM rejected"),
            "requests": counts("SELECT status, COUNT(*) FROM requests GROUP BY status"),
            "snapshots": self.scalar("SELECT COUNT(*) FROM snapshots"),
            "bundles": counts("SELECT status, COUNT(*) FROM bundles GROUP BY status"),
            "high_water": self.high_water(),
        }


class _Transaction:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, *_):
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
