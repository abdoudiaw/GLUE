"""Frozen, content-addressed training snapshots built from the catalog.

A snapshot is a directory `<root>/<snapshot_id>/` holding

* data.nc       - selected input/target variables stacked along `sample`,
                  chunked one sample per chunk, plus per-sample metadata;
* manifest.json - spec, high-water mark, event ids/sha256, splits, exclusions.

The id is a hash of the spec and the selected event sha256s, so rebuilding the
same selection returns the existing snapshot. Snapshots are written to a
temporary directory and renamed into place; they are never modified. Trainers
read snapshots, never the live catalog.

Splits are assigned per case (all events of one SOLPS run share a split) from a
salted hash of case_id, so a case keeps its split as the database grows.

Cases whose every event was requested by active learning were chosen by the
model, so they never enter val/test: the hashed test share of them becomes
`acq_test` (the "replay of difficult calls" set used for promotion) and the
rest is training data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import netCDF4
import numpy as np

from glue2.catalog import Catalog, utcnow
from glue2.events import read_arrays, variable_shapes

SPLITS = ("train", "val", "test", "acq_test")


@dataclass(frozen=True)
class SnapshotSpec:
    inputs: tuple[str, ...]
    targets: tuple[str, ...]
    event_kinds: tuple[str, ...] = ("single_call",)
    used_only: bool = False          # keep only the EIRENE result B2 actually used
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_salt: str = "glue2"
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, d: dict) -> "SnapshotSpec":
        d = dict(d)
        for key in ("inputs", "targets", "event_kinds"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)

    def as_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def hash_split(case_id: str, spec: SnapshotSpec) -> str:
    u = int(hashlib.sha256(f"{spec.split_salt}:{case_id}".encode()).hexdigest()[:12], 16) / 16**12
    if u < spec.test_fraction:
        return "test"
    if u < spec.test_fraction + spec.val_fraction:
        return "val"
    return "train"


def assign_splits(case_origins: dict[str, set[str]], spec: SnapshotSpec) -> dict[str, str]:
    out = {}
    for case, origins in case_origins.items():
        split = hash_split(case, spec)
        if origins == {"al_request"}:
            split = "acq_test" if split == "test" else "train"
        out[case] = split
    return out


@dataclass
class Snapshot:
    path: Path
    manifest: dict = field(repr=False)

    @property
    def snapshot_id(self) -> str:
        return self.manifest["snapshot_id"]

    @property
    def spec(self) -> SnapshotSpec:
        return SnapshotSpec.from_dict(self.manifest["spec"])

    @property
    def shapes(self) -> dict[str, tuple]:
        return {k: tuple(v) for k, v in self.manifest["shapes"].items()}

    @classmethod
    def open(cls, path: str | Path) -> "Snapshot":
        path = Path(path)
        return cls(path, json.loads((path / "manifest.json").read_text()))

    def samples(self, split: str | None = None) -> np.ndarray:
        idx = np.arange(len(self.manifest["events"]))
        if split is None:
            return idx
        return idx[np.array([e["split"] == split for e in self.manifest["events"]], dtype=bool)]

    def load(self, names: list[str] | tuple[str, ...], split: str | None = None) -> dict[str, np.ndarray]:
        idx = self.samples(split)
        out = {}
        with netCDF4.Dataset(self.path / "data.nc") as ds:
            for name in names:
                var = ds.variables[name]
                var.set_auto_mask(False)
                out[name] = np.asarray(var[idx, ...]) if len(idx) else np.empty((0, *var.shape[1:]), var.dtype)
        return out

    def events(self, split: str | None = None) -> list[dict]:
        return [self.manifest["events"][i] for i in self.samples(split)]


def stratum_codes(event_path: str) -> list[str] | None:
    """`['1W', '2E', ...]` from an event's ``b2_crcstra`` (active stratum type codes),
    None when the event does not carry it. Stored in the manifest so a snapshot copied
    to another machine keeps its stratum names without the event files."""
    try:
        import netCDF4
        with netCDF4.Dataset(event_path) as ds:
            codes = ds.variables["b2_crcstra"][...]
    except (OSError, KeyError):
        return None
    codes = [c.decode() if isinstance(c, bytes) else str(c) for c in np.asarray(codes).ravel()]
    return [f"{i + 1}{c.strip()}" for i, c in enumerate(codes)]


def build_snapshot(catalog: Catalog, spec: SnapshotSpec, root: str | Path,
                   high_water: int | None = None) -> Snapshot:
    root = Path(root)
    high_water = catalog.high_water() if high_water is None else high_water
    marks = ",".join("?" for _ in spec.event_kinds)
    used = " AND used_by_b2 = 1" if spec.used_only else ""
    rows = catalog.query(
        f"SELECT * FROM events WHERE event_id <= ? AND event_kind IN ({marks}){used} ORDER BY event_id",
        (high_water, *spec.event_kinds))
    if not rows:
        raise ValueError("no eligible events for snapshot")

    names = spec.inputs + spec.targets
    shapes, selected, excluded = None, [], []
    for row in rows:
        try:
            got = variable_shapes(row["path"], names)
        except (KeyError, OSError) as exc:
            excluded.append({"event_id": row["event_id"], "reason": f"unreadable/missing: {exc}"})
            continue
        if shapes is None:
            shapes = got
        if got != shapes:
            excluded.append({"event_id": row["event_id"], "reason": "shape mismatch with reference event"})
            continue
        selected.append(row)
    if not selected:
        raise ValueError("no events with a consistent variable layout")

    case_origins: dict[str, set[str]] = {}
    for row in selected:
        case_origins.setdefault(row["case_id"], set()).add(row["origin"])
    splits = assign_splits(case_origins, spec)

    events = [{"event_id": r["event_id"], "sha256": r["sha256"], "path": r["path"],
               "case_id": r["case_id"], "background_hash": r["background_hash"],
               "b2_call_index": r["b2_call_index"], "repeat_index": r["repeat_index"],
               "origin": r["origin"], "split": splits[r["case_id"]]} for r in selected]
    ident = json.dumps({"spec": spec.as_dict(), "events": [e["sha256"] for e in events]}, sort_keys=True)
    snapshot_id = hashlib.sha256(ident.encode()).hexdigest()[:16]
    final = root / snapshot_id
    if final.exists():
        existing = Snapshot.open(final)
        with catalog.transaction() as conn:       # register it if a crash beat the insert
            conn.execute("INSERT OR IGNORE INTO snapshots (snapshot_id, path, high_water, n_events, created_utc)"
                         " VALUES (?,?,?,?,?)", (snapshot_id, str(final), existing.manifest["high_water"],
                                                 len(existing.manifest["events"]),
                                                 existing.manifest["created_utc"]))
        return existing

    manifest = {
        "snapshot_id": snapshot_id,
        "created_utc": utcnow(),
        "catalog": str(catalog.path),
        "high_water": high_water,
        "spec": spec.as_dict(),
        "shapes": {k: list(v) for k, v in shapes.items()},
        "counts": {s: sum(e["split"] == s for e in events) for s in SPLITS},
        "cases": {s: sorted({e["case_id"] for e in events if e["split"] == s}) for s in SPLITS},
        "events": events,
        "excluded": excluded,
        "strata": stratum_codes(events[0]["path"]),
    }

    tmp = root / f".{snapshot_id}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        _write_data(tmp / "data.nc", events, names, shapes, spec.dtype)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    with catalog.transaction():
        catalog.insert("snapshots", {"snapshot_id": snapshot_id, "path": str(final),
                                     "high_water": high_water, "n_events": len(events),
                                     "created_utc": manifest["created_utc"]})
    catalog.audit("snapshot", snapshot_id=snapshot_id, n_events=len(events),
                  counts=manifest["counts"], excluded=len(excluded))
    return Snapshot(final, manifest)


def _write_data(path: Path, events: list[dict], names: tuple[str, ...],
                shapes: dict[str, tuple], dtype: str) -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("sample", len(events))
        for name in names:
            dims = [f"{name}_d{i}" for i in range(len(shapes[name]))]
            for dim, size in zip(dims, shapes[name]):
                ds.createDimension(dim, size)
            ds.createVariable(name, dtype, ("sample", *dims), zlib=True, complevel=1,
                              chunksizes=(1, *shapes[name]))
        ds.createVariable("event_id", "i8", ("sample",))[:] = [e["event_id"] for e in events]
        for i, event in enumerate(events):
            arrays = read_arrays(event["path"], names)
            for name in names:
                ds.variables[name][i, ...] = arrays[name].astype(dtype)
