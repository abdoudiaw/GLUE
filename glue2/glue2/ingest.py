"""Register new event files in the catalog.

Files are grouped by case directory. A case written by the training campaign
carries a sha256 manifest and a success marker; such a case is ingested only
once the marker exists, and every event must match its manifest hash. Other
files (teacher answers, ad-hoc copies) are considered once their mtime is
older than `settle_seconds`, because the Fortran writer creates them in place.

Valid files become event rows; invalid ones are recorded in `rejected` and
retried only if they change. Directories matching `exclude` (by default `_*`,
e.g. `_quarantine_*`, and hidden ones) are never scanned. An event whose
background matches a request fulfils it. The sampled run controls in
`source_params.json` are recorded per case.

Event files may be cloud placeholders (e.g. Dropbox online-only): stat() is
free, but reading one downloads it. With `skip_dataless` (default) a case is
left for a later scan until all its files are local; `max_new_cases` bounds
the work of one scan.
"""

from __future__ import annotations

import fnmatch
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from glue2.catalog import Catalog, utcnow
from glue2.events import (CASE_MANIFEST, CASE_PARAMS, CASE_SUCCESS, FILENAME_RE, EventError,
                          case_id_for, read_event_info)

DEFAULT_EXCLUDE = ("_*", ".*")
SF_DATALESS = 0x40000000          # macOS: file content not materialized locally


def is_dataless(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_flags", 0) & SF_DATALESS)


@dataclass
class IngestReport:
    added: list[int] = field(default_factory=list)
    duplicates: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    unsettled: int = 0
    fulfilled: list[str] = field(default_factory=list)
    cases: int = 0
    more: bool = False                    # stopped at max_new_cases

    def as_dict(self) -> dict:
        return {"added": len(self.added), "duplicates": self.duplicates,
                "rejected": len(self.rejected), "unsettled": self.unsettled,
                "fulfilled": self.fulfilled, "cases": self.cases, "more": self.more}


def scan(roots: list[str | Path], exclude: tuple[str, ...] = DEFAULT_EXCLUDE) -> dict[Path, list[Path]]:
    """Event files grouped by their directory, skipping excluded directory names."""
    groups: dict[Path, set[Path]] = {}
    for root in roots:
        root = Path(root)
        if root.is_file():
            groups.setdefault(root.parent.resolve(), set()).add(root.resolve())
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if not any(fnmatch.fnmatch(d, pat) for pat in exclude))
            names = [f for f in filenames if FILENAME_RE.search(f)]
            if names:
                d = Path(dirpath).resolve()
                groups.setdefault(d, set()).update(d / f for f in names)
    return {d: sorted(files) for d, files in sorted(groups.items())}


def case_manifest(case_dir: Path) -> dict[str, str] | None:
    path = case_dir / CASE_MANIFEST
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = digest
    return out


def case_controls(case_dir: Path) -> dict[str, float] | None:
    """Numeric leaves of source_params.json `inputs`, flattened to dotted keys."""
    path = case_dir / CASE_PARAMS
    if not path.exists():
        return None
    try:
        inputs = json.loads(path.read_text()).get("inputs", {})
    except (json.JSONDecodeError, OSError):
        return None
    flat = {}

    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix] = float(node)

    walk(inputs, "")
    return flat


def ingest(catalog: Catalog, roots: list[str | Path], origin: str = "campaign",
           settle_seconds: float = 60.0, now: float | None = None,
           exclude: tuple[str, ...] = DEFAULT_EXCLUDE, max_new_cases: int | None = None,
           skip_dataless: bool = True) -> IngestReport:
    report = IngestReport()
    now = time.time() if now is None else now
    known = catalog.known_paths()
    # Repeats of a requested background may arrive after the request closed.
    requests = {r["background_hash"]: r["request_id"] for r in
                catalog.query("SELECT request_id, background_hash FROM requests WHERE status!='failed'")}

    for case_dir, files in scan(roots, exclude).items():
        pending = []
        for path in files:
            st = os.stat(path)
            if str(path) in known:
                size, mtime = known[str(path)]
                if mtime is None or (size == st.st_size and mtime == st.st_mtime):
                    continue                  # ingested, or rejected and unchanged
            pending.append((path, st))
        if not pending:
            continue

        if skip_dataless and any(is_dataless(st) for _, st in pending):
            report.unsettled += len(pending)  # not downloaded yet
            continue
        manifest = case_manifest(case_dir)
        if manifest is not None and not (case_dir / CASE_SUCCESS).exists():
            report.unsettled += len(pending)  # campaign case still being written
            continue
        if max_new_cases is not None and report.cases >= max_new_cases:
            report.more = True
            break
        report.cases += 1
        _record_case(catalog, case_dir)

        for path, st in pending:
            if manifest is None and now - st.st_mtime < settle_seconds:
                report.unsettled += 1
                continue
            _ingest_file(catalog, path, st, manifest, origin, requests, report)

    if report.added or report.rejected:
        catalog.audit("ingest", roots=[str(r) for r in roots], **report.as_dict())
    return report


def _record_case(catalog: Catalog, case_dir: Path) -> None:
    controls = case_controls(case_dir)
    if controls is not None:
        with catalog.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO cases (case_id, controls, source) VALUES (?,?,?)",
                         (case_id_for(case_dir / "x"), json.dumps(controls, sort_keys=True),
                          str(case_dir / CASE_PARAMS)))


def _ingest_file(catalog: Catalog, path: Path, st: os.stat_result, manifest: dict | None,
                 origin: str, requests: dict, report: IngestReport) -> None:
    key = str(path)

    def reject(reason: str) -> None:
        with catalog.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rejected (path, size, mtime, reason, seen_utc) VALUES (?,?,?,?,?)",
                (key, st.st_size, st.st_mtime, reason, utcnow()))

    try:
        info = read_event_info(path)
        if manifest is not None and manifest.get(path.name) != info.sha256:
            raise EventError("sha256 does not match the case manifest" if path.name in manifest
                             else "file not listed in the case manifest")
    except EventError as exc:
        reject(str(exc))
        report.rejected.append((key, str(exc)))
        return
    existing = catalog.scalar("SELECT event_id FROM events WHERE sha256=?", (info.sha256,))
    if existing:
        reject(f"duplicate of event {existing}")  # same bytes under another path
        report.duplicates += 1
        return

    request_id = requests.get(info.background_hash)
    row = info.as_row() | {
        "origin": "al_request" if request_id else origin,
        "request_id": request_id,
        "ingested_utc": utcnow(),
    }
    with catalog.transaction() as conn:
        conn.execute("DELETE FROM rejected WHERE path=?", (key,))
        report.added.append(catalog.insert("events", row))
        if request_id:
            cur = conn.execute(
                "UPDATE requests SET status='fulfilled', closed_utc=? WHERE request_id=? AND status='open'",
                (utcnow(), request_id))
            if cur.rowcount:
                report.fulfilled.append(request_id)
