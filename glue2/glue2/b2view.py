"""Training arrays from a snapshot, trimmed to what B2.5 feeds back.

A snapshot stores whole event variables: EIRBRA arrays with all stratum
slots (including unused ones), three momentum slots, and the guard cells of
the B2 mesh. This module turns one snapshot into dense arrays for a learner:

- ``inputs``   (sample, channel, ny, nx)  interior cells of the BRAEIR fields
- ``targets``  (sample, channel, ny, nx)  interior cells of the four sources,
  one channel per active stratum (particle, parallel momentum, electron
  energy, ion energy), in the raw per-stratum units EIRENE returns
- ``weights``  (sample, stratum)          tflux * flux_scale, the factor B2.5
  applies to each stratum
- ``tallies``  {name: (sample, surface)}  stratum-total wall tallies that B2.5
  turns into the core-boundary neutral flux

Schema-3 events store BRAEIR on the B2 mesh with guard cells (``bra_x =
nx + 2``) and EIRBRA with two further scratch columns; both are indexed as
``[iy + 1, ix + 1]``, so the same slice selects the interior of each.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glue2.snapshot import Snapshot

SOURCES = ("eirbra_sni", "eirbra_smo", "eirbra_see", "eirbra_sei")
CONTROLS = ("b2_tflux", "b2_flux_scale")
TALLY_PREFIX = "wneutrals_wld"


@dataclass
class TrainingArrays:
    inputs: np.ndarray
    input_names: list[str]
    targets: np.ndarray
    target_names: list[str]
    weights: np.ndarray
    tallies: dict[str, np.ndarray]
    strata: list[str]
    event_ids: np.ndarray
    cases: list[str]
    split: str | None = None
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_strata(self) -> int:
        return len(self.strata)

    def source_sum(self, name: str) -> np.ndarray:
        """``sum(raw * tflux * flux_scale)`` over strata, as B2.5 forms it: (sample, ny, nx)."""
        channels = [i for i, n in enumerate(self.target_names) if n.startswith(name + ":")]
        return np.einsum("ns,nsyx->nyx", self.weights, self.targets[:, channels])


def _interior(arr: np.ndarray, nx_bra: int) -> np.ndarray:
    """Interior cells of a (..., y, x) array stored with B2 guard cells."""
    return arr[..., 1:-1, 1 : nx_bra - 1]


def training_arrays(snapshot: Snapshot, split: str | None = None,
                    strata: list[str] | None = None) -> TrainingArrays:
    """Dense learner arrays from a snapshot built with the v3 spec.

    ``strata`` are the stratum type codes to keep as targets (default: all
    active strata, in order). The codes come from the events' ``b2_crcstra``
    when the snapshot carries it, else they are numbered.
    """
    spec = snapshot.spec
    shapes = snapshot.shapes
    missing = [n for n in SOURCES + CONTROLS if n not in spec.inputs + spec.targets]
    if missing:
        raise ValueError(f"snapshot spec lacks {missing}")
    bra = [n for n in spec.inputs if n.startswith("braeir_") and len(shapes[n]) >= 2]
    if not bra:
        raise ValueError("snapshot has no BRAEIR cell fields among its inputs")
    nx_bra = shapes[bra[0]][-1]
    ny_bra = shapes[bra[0]][-2]
    if shapes["eirbra_see"][-2] != ny_bra or shapes["eirbra_see"][-1] < nx_bra:
        raise ValueError("EIRBRA and BRAEIR layouts disagree; is this a schema-2 snapshot?")

    names = list(spec.inputs + spec.targets)
    data = snapshot.load(names, split)
    n = len(snapshot.samples(split))
    events = snapshot.events(split)

    # inputs: every BRAEIR cell field, first species where there is a species axis
    inputs, input_names = [], []
    for name in bra:
        arr = data[name]
        while arr.ndim > 3:
            arr = arr[:, 0]
        inputs.append(_interior(arr, nx_bra))
        input_names.append(name.split("_", 1)[1])
    inputs = np.stack(inputs, axis=1) if inputs else np.empty((n, 0, ny_bra - 2, nx_bra - 2))

    # strata: active ones are those with a control value
    n_active = shapes["b2_tflux"][0]
    labels = _stratum_labels(snapshot, n_active)
    keep = list(range(n_active)) if strata is None else [labels.index(s) for s in strata]
    weights = (data["b2_tflux"] * data["b2_flux_scale"])[:, keep].astype(np.float32)

    # targets: per-stratum raw sources, parallel momentum only
    targets, target_names = [], []
    for name in SOURCES:
        arr = data[name][:, keep]                   # (n, strata, [species|slot], y, x)
        if arr.ndim == 5:
            arr = arr[:, :, 0]
        arr = _interior(arr, nx_bra)
        for j, k in enumerate(keep):
            targets.append(arr[:, j])
            target_names.append(f"{name.split('_', 1)[1]}:{labels[k]}")
    targets = np.stack(targets, axis=1)

    tallies = {}
    for name in spec.targets:
        if name.startswith(TALLY_PREFIX) and name in data:
            arr = data[name]
            if arr.ndim == 4:                       # (n, stratum slot, species, surface)
                arr = arr[:, 0, 0]
            elif arr.ndim == 3:
                arr = arr[:, 0]
            tallies[name.split("_", 1)[1]] = arr

    return TrainingArrays(
        inputs=inputs.astype(np.float32), input_names=input_names,
        targets=targets.astype(np.float32), target_names=target_names,
        weights=weights, tallies=tallies, strata=[labels[k] for k in keep],
        event_ids=np.array([e["event_id"] for e in events]), cases=[e["case_id"] for e in events],
        split=split,
    )


def _stratum_labels(snapshot: Snapshot, n_active: int) -> list[str]:
    """``1W, 2E, ...`` from the first event's ``b2_crcstra``, else ``1, 2, ...``."""
    try:
        import netCDF4
        with netCDF4.Dataset(snapshot.manifest["events"][0]["path"]) as ds:
            codes = ds.variables["b2_crcstra"][...]
        codes = [c.decode() if isinstance(c, bytes) else str(c) for c in np.asarray(codes).ravel()]
        if len(codes) == n_active:
            return [f"{i + 1}{c.strip()}" for i, c in enumerate(codes)]
    except (OSError, KeyError):
        pass
    return [str(i + 1) for i in range(n_active)]
