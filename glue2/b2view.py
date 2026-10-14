"""Training arrays from a snapshot, trimmed to what B2.5 feeds back.

A snapshot stores whole event variables: EIRBRA arrays with all stratum
slots (including unused ones), three momentum slots, and the guard cells of
the B2 mesh. This module turns one snapshot into dense arrays for a learner
and maps a learner's output back into event-shaped arrays:

- ``inputs``   (sample, channel, ny, nx)  interior cells of the BRAEIR fields
- ``targets``  (sample, channel, ny, nx)  interior cells of the sources, one
  channel per (kind, active stratum): particle ``sni``, parallel momentum
  ``smo`` (and, when asked for, the radial ``smr`` and diamagnetic ``smd``
  momentum slots that B2.5 uses only with drifts), electron energy ``see``,
  ion energy ``sei``, in the raw per-stratum units EIRENE returns
- ``weights``  (sample, stratum)          tflux * flux_scale, the factor B2.5
  applies to each stratum
- ``tallies``  {name: (sample, stratum, surface)}  per-stratum wall tallies.
  B2.5 reads them stratum by stratum (``wldna(j, is, istrai)`` for
  ``istrai = 1..nstrat``) when it renormalises the sources for particle
  balance; slot 0 of the event array is the sum over strata.

Schema-3 events store BRAEIR on the B2 mesh with guard cells (``bra_x =
nx + 2``) and EIRBRA with two further scratch columns; both are indexed as
``[iy + 1, ix + 1]``, so the same slice selects the interior of each.
EIRBRA stratum slot ``k`` and wall-tally slot ``k + 1`` both hold stratum
``k + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glue2.snapshot import Snapshot

SOURCES = ("eirbra_sni", "eirbra_smo", "eirbra_see", "eirbra_sei")
CONTROLS = ("b2_tflux", "b2_flux_scale")
TALLY_PREFIX = "wneutrals_wld"
MOMENTUM_KINDS = ("smo", "smr", "smd")      # eirbra_smo momentum slots 0, 1, 2


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
    layout: "EventLayout | None" = None
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


@dataclass
class EventLayout:
    """Where a learner's channels live in the event arrays of one snapshot spec.

    Built once from a snapshot; a bundle stores it (``as_dict``) so that
    ``predict`` can return event-shaped arrays without the snapshot."""
    shapes: dict[str, tuple]          # event variable -> shape without the sample axis
    bra_names: list[str]              # BRAEIR variables used as inputs, in order
    keep: list[int]                   # active stratum indices kept as targets
    labels: list[str]                 # labels of all active strata
    momentum_slots: tuple[int, ...]
    tally_names: list[str]            # event variables of the per-stratum wall tallies

    @property
    def nx_bra(self) -> int:
        return self.shapes[self.bra_names[0]][-1]

    @property
    def strata(self) -> list[str]:
        return [self.labels[k] for k in self.keep]

    @property
    def input_names(self) -> list[str]:
        return [n.split("_", 1)[1] for n in self.bra_names]

    @property
    def target_names(self) -> list[str]:
        out = []
        for name in SOURCES:
            kinds = [MOMENTUM_KINDS[s] for s in self.momentum_slots] if name == "eirbra_smo" \
                else [name.split("_", 1)[1]]
            for kind in kinds:
                out += [f"{kind}:{self.labels[k]}" for k in self.keep]
        return out

    def as_dict(self) -> dict:
        return {"shapes": {k: list(v) for k, v in self.shapes.items()}, "bra_names": self.bra_names,
                "keep": self.keep, "labels": self.labels, "momentum_slots": list(self.momentum_slots),
                "tally_names": self.tally_names}

    @classmethod
    def from_dict(cls, d: dict) -> "EventLayout":
        return cls(shapes={k: tuple(v) for k, v in d["shapes"].items()}, bra_names=list(d["bra_names"]),
                   keep=list(d["keep"]), labels=list(d["labels"]),
                   momentum_slots=tuple(d["momentum_slots"]), tally_names=list(d["tally_names"]))

    # ---- event arrays -> learner arrays --------------------------------------------

    def pack_inputs(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """(sample, channel, ny, nx) from event-shaped BRAEIR arrays (first species
        where there is a species axis)."""
        cols = []
        for name in self.bra_names:
            arr = np.asarray(data[name])
            while arr.ndim > 3:
                arr = arr[:, 0]
            cols.append(_interior(arr, self.nx_bra))
        return np.stack(cols, axis=1).astype(np.float32)

    def pack_weights(self, data: dict[str, np.ndarray]) -> np.ndarray:
        return (np.asarray(data["b2_tflux"]) * np.asarray(data["b2_flux_scale"]))[:, self.keep].astype(np.float32)

    def pack_targets(self, data: dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for name in SOURCES:
            arr = np.asarray(data[name])[:, self.keep]        # (n, strata, [species|slot], y, x)
            slots = self.momentum_slots if name == "eirbra_smo" else (0,)
            for s in slots:
                a = arr[:, :, s] if arr.ndim == 5 else arr
                cols += [_interior(a, self.nx_bra)[:, j] for j in range(len(self.keep))]
        return np.stack(cols, axis=1).astype(np.float32)

    def pack_tallies(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """{name: (sample, stratum, surface)} from (sample, slot, species, surface)."""
        out = {}
        for name in self.tally_names:
            arr = np.asarray(data[name])
            if arr.ndim == 4:
                arr = arr[:, :, 0]
            out[name.split("_", 1)[1]] = arr[:, [k + 1 for k in self.keep]].astype(np.float32)
        return out

    # ---- learner arrays -> event arrays --------------------------------------------

    def unpack(self, targets: np.ndarray, tallies: dict[str, np.ndarray] | None = None,
               dtype=np.float64) -> dict[str, np.ndarray]:
        """Event-shaped arrays (sample, *shape) from learner outputs: guard cells,
        unused strata and momentum slots not predicted are zero; tally slot 0 is
        the sum over the predicted strata."""
        n = len(targets)
        names = self.target_names
        out = {}
        for name in SOURCES:
            full = np.zeros((n, *self.shapes[name]), dtype)
            kinds = [MOMENTUM_KINDS[s] for s in self.momentum_slots] if name == "eirbra_smo" \
                else [name.split("_", 1)[1]]
            for kind in kinds:
                slot = MOMENTUM_KINDS.index(kind) if name == "eirbra_smo" else 0
                for j, k in enumerate(self.keep):
                    ch = targets[:, names.index(f"{kind}:{self.labels[k]}")]
                    view = full[:, k, slot] if full.ndim == 5 else full[:, k]
                    view[:, 1:-1, 1 : self.nx_bra - 1] = ch
            out[name] = full
        for name in self.tally_names:
            full = np.zeros((n, *self.shapes[name]), dtype)
            arr = (tallies or {})[name.split("_", 1)[1]]
            for j, k in enumerate(self.keep):
                (full[:, k + 1, 0] if full.ndim == 4 else full[:, k + 1])[:] = arr[:, j]
            (full[:, 0, 0] if full.ndim == 4 else full[:, 0])[:] = arr.sum(1)
            out[name] = full
        return out


def event_layout(snapshot: Snapshot, strata: list[str] | None = None,
                 momentum_slots: tuple[int, ...] = (0,)) -> EventLayout:
    spec = snapshot.spec
    shapes = snapshot.shapes
    missing = [n for n in SOURCES + CONTROLS if n not in spec.inputs + spec.targets]
    if missing:
        raise ValueError(f"snapshot spec lacks {missing}")
    bra = [n for n in spec.inputs if n.startswith("braeir_") and len(shapes[n]) >= 2]
    if not bra:
        raise ValueError("snapshot has no BRAEIR cell fields among its inputs")
    nx_bra, ny_bra = shapes[bra[0]][-1], shapes[bra[0]][-2]
    if shapes["eirbra_see"][-2] != ny_bra or shapes["eirbra_see"][-1] < nx_bra:
        raise ValueError("EIRBRA and BRAEIR layouts disagree; is this a schema-2 snapshot?")
    if any(s not in range(len(MOMENTUM_KINDS)) for s in momentum_slots):
        raise ValueError(f"momentum_slots must be among {list(range(len(MOMENTUM_KINDS)))}")
    n_active = shapes["b2_tflux"][0]
    labels = _stratum_labels(snapshot, n_active)
    keep = list(range(n_active)) if strata is None else [labels.index(s) for s in strata]
    tally_names = [n for n in spec.targets if n.startswith(TALLY_PREFIX)]
    for name in tally_names:
        if shapes[name][0] < n_active + 1:
            raise ValueError(f"{name} has {shapes[name][0]} stratum slots, need {n_active + 1}")
    return EventLayout(shapes=shapes, bra_names=bra, keep=keep, labels=labels,
                       momentum_slots=tuple(momentum_slots), tally_names=tally_names)


def training_arrays(snapshot: Snapshot, split: str | None = None,
                    strata: list[str] | None = None,
                    momentum_slots: tuple[int, ...] = (0,)) -> TrainingArrays:
    """Dense learner arrays from a snapshot built with the v3 spec.

    ``strata`` are the stratum type codes to keep as targets (default: all
    active strata, in order). The codes come from the events' ``b2_crcstra``
    when the snapshot carries it, else they are numbered. ``momentum_slots``
    selects the eirbra_smo slots to learn (0 parallel, 1 radial, 2 diamagnetic).
    """
    layout = event_layout(snapshot, strata, momentum_slots)
    spec = snapshot.spec
    data = snapshot.load(list(spec.inputs + spec.targets), split)
    events = snapshot.events(split)
    return TrainingArrays(
        inputs=layout.pack_inputs(data), input_names=layout.input_names,
        targets=layout.pack_targets(data), target_names=layout.target_names,
        weights=layout.pack_weights(data), tallies=layout.pack_tallies(data), strata=layout.strata,
        event_ids=np.array([e["event_id"] for e in events]), cases=[e["case_id"] for e in events],
        split=split, layout=layout,
    )


def _stratum_labels(snapshot: Snapshot, n_active: int) -> list[str]:
    """``1W, 2E, ...`` from the manifest, else from the first event's ``b2_crcstra``,
    else ``1, 2, ...``."""
    from glue2.snapshot import stratum_codes
    codes = snapshot.manifest.get("strata") or stratum_codes(snapshot.manifest["events"][0]["path"])
    if codes and len(codes) == n_active:
        return list(codes)
    return [str(i + 1) for i in range(n_active)]
