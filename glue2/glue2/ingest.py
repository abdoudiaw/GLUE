"""Register new event files in the catalog.

The Fortran writer creates files in place, so a file is only considered once
its mtime is older than `settle_seconds`. Valid files become event rows;
invalid ones are recorded in `rejected` and retried only if they change.
An event whose background matches an open request fulfils that request.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from glue2.catalog import Catalog, utcnow
from glue2.events import EVENT_GLOB, EventError, read_event_info


@dataclass
class IngestReport:
    added: list[int] = field(default_factory=list)
    duplicates: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    unsettled: int = 0
    fulfilled: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"added": len(self.added), "duplicates": self.duplicates,
                "rejected": len(self.rejected), "unsettled": self.unsettled,
                "fulfilled": self.fulfilled}


def scan(roots: list[str | Path]) -> list[Path]:
    files = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            files.add(root.resolve())
        elif root.is_dir():
            files.update(p.resolve() for p in root.rglob(EVENT_GLOB))
    return sorted(files)


def ingest(catalog: Catalog, roots: list[str | Path], origin: str = "campaign",
           settle_seconds: float = 60.0, now: float | None = None) -> IngestReport:
    report = IngestReport()
    now = time.time() if now is None else now
    known = catalog.known_paths()
    # Repeats of a requested background may arrive after the request closed.
    requests = {r["background_hash"]: r["request_id"] for r in
                catalog.query("SELECT request_id, background_hash FROM requests WHERE status!='failed'")}

    for path in scan(roots):
        key = str(path)
        st = os.stat(path)
        if key in known:
            size, mtime = known[key]
            if mtime is None or (size == st.st_size and mtime == st.st_mtime):
                continue                              # ingested, or rejected and unchanged
        if now - st.st_mtime < settle_seconds:
            report.unsettled += 1
            continue
        def reject(reason: str) -> None:
            with catalog.transaction():
                catalog.conn.execute(
                    "INSERT OR REPLACE INTO rejected (path, size, mtime, reason, seen_utc) VALUES (?,?,?,?,?)",
                    (key, st.st_size, st.st_mtime, reason, utcnow()))

        try:
            info = read_event_info(path)
        except EventError as exc:
            reject(str(exc))
            report.rejected.append((key, str(exc)))
            continue
        existing = catalog.scalar("SELECT event_id FROM events WHERE sha256=?", (info.sha256,))
        if existing:
            reject(f"duplicate of event {existing}")  # same bytes under another path
            report.duplicates += 1
            continue

        request_id = requests.get(info.background_hash)
        row = info.as_row() | {
            "origin": "al_request" if request_id else origin,
            "request_id": request_id,
            "ingested_utc": utcnow(),
        }
        with catalog.transaction():
            catalog.conn.execute("DELETE FROM rejected WHERE path=?", (key,))
            report.added.append(catalog.insert("events", row))
            if request_id:
                cur = catalog.conn.execute(
                    "UPDATE requests SET status='fulfilled', closed_utc=? WHERE request_id=? AND status='open'",
                    (utcnow(), request_id))
                if cur.rowcount:
                    report.fulfilled.append(request_id)

    if report.added or report.rejected:
        catalog.audit("ingest", roots=[str(r) for r in roots], **report.as_dict())
    return report
