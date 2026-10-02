"""Read and validate SOLPS-ITER EIRENE training events (schema v2).

An event is one NetCDF file written by b2mod_eirene_training_dump at the
eirene_eirsrt seam: BRAEIR inputs, b2_* call controls, and the EIRBRA,
WNEUTRALS and CESTIM return state. Event files are immutable; everything
here only reads them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import netCDF4
import numpy as np

SCHEMA_NAME = "solps_eirene_training_event"
SUPPORTED_MAJOR = "2"
EVENT_GLOB = "eirene_training_v*_b2call_*.nc"
FILENAME_RE = re.compile(
    r"eirene_training_v(?P<v>\d+)_b2call_(?P<call>\d{8})_"
    r"(?P<kind>single_call|average_used_by_b2)_(?P<rep>\d{4})\.nc$"
)
CASE_DIR_RE = re.compile(r"^run_[^/]+$")

# Per-case sidecar files written by the training campaign.
CASE_MANIFEST = "eirene_training_v2.sha256"
CASE_SUCCESS = "EIRENE_TRAINING_SUCCESS"
CASE_PARAMS = "source_params.json"

# Minimum content for a usable event; the full inventory is configuration-dependent.
REQUIRED_INPUTS = ("braeir_dni", "braeir_te", "braeir_ti", "braeir_vol")
REQUIRED_CONTROLS = ("b2_tflux", "b2_flux_scale")
REQUIRED_OUTPUTS = ("eirbra_sni", "eirbra_smo", "eirbra_see", "eirbra_sei")

# Prefixes that define the plasma background seen by EIRENE. Repeated EIRENE
# calls for one B2 call share these exactly, so they identify the background.
BACKGROUND_PREFIXES = ("braeir_", "b2_")
# EIRENE index-maps DELTA_SHEATH[XY]B in place (eirmod_infcop.F) and B2 does not
# refresh them before a repeated call, so they differ between repeats of one
# background. They are excluded from the identity (and should not be model inputs
# until that is resolved).
BACKGROUND_EXCLUDE = ("braeir_delta_sheathx", "braeir_delta_sheathy")
PROVENANCE_ATTRS = ("solps_iter_git", "eirene_git", "b2_5_git")


class EventError(ValueError):
    """The file is not a valid training event."""


@dataclass(frozen=True)
class EventInfo:
    path: str
    sha256: str
    size: int
    case_id: str
    b2_call_index: int
    event_kind: str
    repeat_index: int
    repeat_count: int
    background_hash: str
    schema_version: str
    created_local: str
    solps_iter_git: str
    eirene_git: str
    b2_5_git: str

    def as_row(self) -> dict:
        return asdict(self)


def file_sha256(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def case_id_for(path: str | Path) -> str:
    """Nearest enclosing run_* directory, else the parent directory name."""
    path = Path(path).resolve()
    for parent in path.parents:
        if CASE_DIR_RE.match(parent.name):
            return parent.name
    return path.parent.name


def _float_array(var) -> np.ndarray | None:
    if var.dtype.kind not in "fiu":
        return None
    var.set_auto_mask(False)
    return np.asarray(var[...], dtype=np.float64)


def background_hash(ds: netCDF4.Dataset) -> str:
    """Hash of the BRAEIR fields and b2_* call controls, independent of file layout."""
    h = hashlib.sha256()
    for name in sorted(ds.variables):
        if not name.startswith(BACKGROUND_PREFIXES) or name in BACKGROUND_EXCLUDE:
            continue
        var = ds.variables[name]
        var.set_auto_mask(False)
        arr = np.ascontiguousarray(var[...])
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.astype(arr.dtype.newbyteorder("<")).tobytes())
    return h.hexdigest()


def validate(ds: netCDF4.Dataset) -> None:
    schema = getattr(ds, "schema_name", None)
    if schema != SCHEMA_NAME:
        raise EventError(f"schema_name is {schema!r}, expected {SCHEMA_NAME!r}")
    version = str(getattr(ds, "schema_version", ""))
    if version.split(".")[0] != SUPPORTED_MAJOR:
        raise EventError(f"unsupported schema_version {version!r}")
    missing = [n for n in REQUIRED_INPUTS + REQUIRED_CONTROLS + REQUIRED_OUTPUTS
               if n not in ds.variables]
    if missing:
        raise EventError(f"missing variables: {missing}")
    for name, var in ds.variables.items():
        arr = _float_array(var)
        if arr is not None and not np.isfinite(arr).all():
            raise EventError(f"{name} has non-finite values")


def read_event_info(path: str | Path, case_id: str | None = None) -> EventInfo:
    """Validate one event file and return its catalog metadata."""
    path = Path(path)
    match = FILENAME_RE.search(path.name)
    if not match:
        raise EventError(f"unrecognized event filename {path.name!r}")
    try:
        ds = netCDF4.Dataset(path, "r")
    except OSError as exc:
        raise EventError(f"cannot open: {exc}") from exc
    with ds:
        validate(ds)
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        bg = background_hash(ds)
    kind = str(attrs.get("event_kind", match["kind"]))
    if kind != match["kind"]:
        raise EventError(f"event_kind {kind!r} disagrees with filename")
    return EventInfo(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        size=path.stat().st_size,
        case_id=case_id or case_id_for(path),
        b2_call_index=int(attrs.get("b2_call_index", int(match["call"]))),
        event_kind=kind,
        repeat_index=int(attrs.get("eirene_repeat_index", int(match["rep"]))),
        repeat_count=int(attrs.get("eirene_repeat_count", 1)),
        background_hash=bg,
        schema_version=str(attrs["schema_version"]),
        created_local=str(attrs.get("created_local", "")),
        **{k: str(attrs.get(k, "")) for k in PROVENANCE_ATTRS},
    )


def read_arrays(path: str | Path, names: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read named numeric variables as float64 arrays; missing names raise KeyError."""
    out = {}
    with netCDF4.Dataset(path, "r") as ds:
        for name in names:
            arr = _float_array(ds.variables[name])
            if arr is None:
                raise EventError(f"{name} is not numeric")
            out[name] = arr
    return out


def variable_shapes(path: str | Path, names: list[str] | tuple[str, ...]) -> dict[str, tuple]:
    with netCDF4.Dataset(path, "r") as ds:
        return {n: tuple(ds.variables[n].shape) for n in names}
