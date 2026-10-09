"""Utilities for reading and plotting SOLPS-ITER ``fort.31`` files.

The parser follows ``write_f31``/``GFSUB3`` in the B2.5 revision pinned by
abdoudiaw/SOLPS-ITER (cb7d15b2878074bb450666bebee4b1df02ef79e6).
``fort.31`` contains no header, so callers must supply the mesh size.  The
reader validates the total record count and infers the number of charged
fluids (``NFLAI``) from the pinned writer layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np
import pandas as pd
import xarray as xr


ELEMENTARY_CHARGE = 1.602176634e-19


@dataclass(frozen=True)
class FieldSpec:
    name: str
    components: int
    units: str
    description: str
    signed: bool = False
    writer_placeholder: bool = False


def fort31_schema(nflai: int) -> list[FieldSpec]:
    """Return the exact record sequence written by the pinned B2.5 writer."""
    n = int(nflai)
    if n < 1:
        raise ValueError(f"NFLAI must be positive, got {nflai!r}")

    return [
        FieldSpec("DNIB", n, "m^-3", "charged-fluid density"),
        FieldSpec("UUB", n, "m s^-1", "poloidal velocity", signed=True),
        FieldSpec("VVB", n, "m s^-1", "radial velocity", signed=True),
        FieldSpec("WWB", n, "m s^-1", "toroidal velocity", signed=True),
        FieldSpec("TEB", 1, "J", "electron temperature (raw B2/EIRENE unit)"),
        FieldSpec("TIB", 1, "J", "ion temperature (raw B2/EIRENE unit)"),
        FieldSpec("PRB", 1, "Pa", "plasma pressure"),
        FieldSpec("UPB", n, "m s^-1", "parallel velocity", signed=True),
        FieldSpec("RRB", 1, "1", "pitch quantity, documented as Bx/Btot", signed=True),
        FieldSpec("FNIXB", n, "s^-1", "poloidal ion flow on left face", signed=True),
        FieldSpec("FNIYB", n, "s^-1", "radial ion flow on bottom face", signed=True),
        FieldSpec("FEIXB", 1, "W", "poloidal ion heat flow on left face", signed=True),
        FieldSpec("FEIYB", 1, "W", "radial ion heat flow on bottom face", signed=True),
        FieldSpec("FEEXB", 1, "W", "poloidal electron heat flow on left face", signed=True),
        FieldSpec("FEEYB", 1, "W", "radial electron heat flow on bottom face", signed=True),
        FieldSpec("UUDIAB", n, "m s^-1", "total ion drift velocity, diamagnetic direction", signed=True),
        FieldSpec("VVDIAB", n, "m s^-1", "total ion drift velocity, radial direction", signed=True),
        FieldSpec("POB", 1, "V", "electric potential"),
        FieldSpec("VOLB", 1, "m^3", "cell volume"),
        FieldSpec("BFELDB", 1, "T", "magnetic-field magnitude"),
        FieldSpec("BPOLB", 1, "T", "poloidal magnetic field", signed=True),
        FieldSpec("BRADB", 1, "T", "radial magnetic field", signed=True),
        FieldSpec("BTORB", 1, "T", "toroidal magnetic field", signed=True),
        FieldSpec("VPARXB", n, "placeholder", "future-use parallel x velocity", signed=True, writer_placeholder=True),
        FieldSpec("VPARYB", n, "placeholder", "future-use parallel y velocity", signed=True, writer_placeholder=True),
        FieldSpec("VRADXB", n, "placeholder", "future-use radial x velocity", signed=True, writer_placeholder=True),
        FieldSpec("VRADYB", n, "placeholder", "future-use radial y velocity", signed=True, writer_placeholder=True),
        FieldSpec("DELTAE_PARXB", 1, "placeholder", "future-use electron parallel x correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTAE_PARYB", 1, "placeholder", "future-use electron parallel y correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTAE_RADXB", 1, "placeholder", "future-use electron radial x correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTAE_RADYB", 1, "placeholder", "future-use electron radial y correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTAI_PARXB", 1, "placeholder", "future-use ion parallel x correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTAI_PARYB", 1, "placeholder", "future-use ion parallel y correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTAI_RADXB", 1, "placeholder", "future-use ion radial x correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTAI_RADYB", 1, "placeholder", "future-use ion radial y correction", signed=True, writer_placeholder=True),
        FieldSpec("DELTA_SHEATHXB", 1, "implementation-specific", "x-face sheath correction", signed=True),
        FieldSpec("DELTA_SHEATHYB", 1, "implementation-specific", "y-face sheath correction", signed=True),
        FieldSpec("ZIB", n, "1", "average charged-fluid charge"),
    ]


def infer_nflai(n_values: int, nx: int, ny: int) -> tuple[int, int]:
    """Infer ``NFLAI`` and the number of 2-D records.

    The pinned writer emits ``14*NFLAI + 24`` records, each of shape
    ``(ny+2, nx+2)``.  ``AISOB`` exists in memory but is not written.
    """
    n_cells = (int(nx) + 2) * (int(ny) + 2)
    if n_values % n_cells:
        raise ValueError(
            f"fort.31 has {n_values:,} values, not a multiple of "
            f"(nx+2)*(ny+2)={n_cells:,}"
        )
    n_records = n_values // n_cells
    remainder = n_records - 24
    if remainder <= 0 or remainder % 14:
        raise ValueError(
            f"{n_records} records do not match the pinned 14*NFLAI+24 layout"
        )
    return remainder // 14, n_records


def read_fort31(path: str | Path, nx: int, ny: int) -> dict:
    """Read a headerless ``fort.31`` into named ``(component, y, x)`` arrays."""
    path = Path(path)
    values = np.fromfile(path, dtype=float, sep=" ")
    nflai, n_records = infer_nflai(values.size, nx, ny)
    n_cells = (nx + 2) * (ny + 2)
    schema = fort31_schema(nflai)

    fields: dict[str, np.ndarray] = {}
    cursor = 0
    for spec in schema:
        count = spec.components * n_cells
        block = values[cursor : cursor + count]
        if block.size != count:
            raise ValueError(f"truncated field {spec.name}")
        # GFSUB3 writes IS, then IY, then IX.  C-order reshape reproduces that
        # stream as (component, y, x).
        fields[spec.name] = block.reshape(spec.components, ny + 2, nx + 2)
        cursor += count
    if cursor != values.size:
        raise ValueError(f"parser consumed {cursor:,} of {values.size:,} values")

    return {
        "path": path,
        "nx": int(nx),
        "ny": int(ny),
        "nflai": int(nflai),
        "n_records": int(n_records),
        "n_values": int(values.size),
        "schema": schema,
        "fields": fields,
    }


def load_balance_geometry(path: str | Path) -> dict[str, np.ndarray | int]:
    """Load mesh dimensions and cell vertices from ``balance.nc``."""
    with xr.open_dataset(path) as ds:
        nx = int(ds.sizes["nx_plus2"] - 2)
        ny = int(ds.sizes["ny_plus2"] - 2)
        return {
            "nx": nx,
            "ny": ny,
            "crx": np.asarray(ds["crx"].values),
            "cry": np.asarray(ds["cry"].values),
            "vol": np.asarray(ds["vol"].values),
            "za": np.asarray(ds["za"].values),
            "species": np.asarray(ds["species"].values),
            "te": np.asarray(ds["te"].values),
            "ti": np.asarray(ds["ti"].values),
            "na": np.asarray(ds["na"].values),
        }


def summary_table(parsed: Mapping, include_guards: bool = False) -> pd.DataFrame:
    """Return one statistics row per field component."""
    rows = []
    fields = parsed["fields"]
    for spec in parsed["schema"]:
        for component, plane in enumerate(fields[spec.name]):
            values = plane if include_guards else plane[1:-1, 1:-1]
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "field": spec.name,
                    "component": component,
                    "units": spec.units,
                    "description": spec.description,
                    "writer_placeholder": spec.writer_placeholder,
                    "all_zero": bool(np.all(finite == 0)) if finite.size else True,
                    "min": float(np.min(finite)) if finite.size else np.nan,
                    "median": float(np.median(finite)) if finite.size else np.nan,
                    "max": float(np.max(finite)) if finite.size else np.nan,
                    "max_abs": float(np.max(np.abs(finite))) if finite.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def validate_against_balance(parsed: Mapping, geometry: Mapping) -> pd.DataFrame:
    """Check the parser order against charged density and temperature in balance.nc."""
    fields = parsed["fields"]
    za = np.asarray(geometry["za"])
    charged_indices = np.flatnonzero(za > 0)
    checks = []

    for component, species_index in enumerate(charged_indices[: parsed["nflai"]]):
        checks.append((f"DNIB[{component}] vs na[{species_index}]", fields["DNIB"][component], geometry["na"][species_index]))
    checks.extend(
        [
            ("TEB vs te", fields["TEB"][0], geometry["te"]),
            ("TIB vs ti", fields["TIB"][0], geometry["ti"]),
        ]
    )
    rows = []
    for name, actual, reference in checks:
        a = np.asarray(actual)[1:-1, 1:-1]
        b = np.asarray(reference)[1:-1, 1:-1]
        diff = a - b
        scale = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
        rows.append(
            {
                "check": name,
                "max_abs_difference": float(np.max(np.abs(diff))),
                "max_difference/reference_max": float(np.max(np.abs(diff)) / scale),
                "correlation": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def display_plane(parsed: Mapping, name: str, component: int = 0) -> tuple[np.ndarray, str]:
    """Return an interior plane and display unit, converting temperatures to eV."""
    plane = np.asarray(parsed["fields"][name][component, 1:-1, 1:-1], dtype=float)
    spec = next(item for item in parsed["schema"] if item.name == name)
    if name in {"TEB", "TIB"}:
        return plane / ELEMENTARY_CHARGE, "eV"
    return plane, spec.units


def cell_polygons(geometry: Mapping) -> np.ndarray:
    """Return interior SOLPS cell polygons with shape ``(ncell, 4, 2)``."""
    crx = np.asarray(geometry["crx"])[:, 1:-1, 1:-1]
    cry = np.asarray(geometry["cry"])[:, 1:-1, 1:-1]
    # B2 vertex order is not perimeter order.  0 -> 1 -> 3 -> 2 is cyclic.
    order = [0, 1, 3, 2]
    vertices = np.stack((crx[order], cry[order]), axis=-1)
    return np.moveaxis(vertices, 0, 2).reshape(-1, 4, 2)


def _norm_and_cmap(values: np.ndarray, signed: bool):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(0, 1), "viridis"
    if signed or (np.min(finite) < 0 < np.max(finite)):
        limit = float(np.nanpercentile(np.abs(finite), 99.5))
        if limit == 0:
            limit = 1.0
        positive = np.abs(finite[np.nonzero(finite)])
        linthresh = max(float(np.nanpercentile(positive, 10)) if positive.size else 0.0, limit * 1.0e-4)
        return SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit, base=10), "coolwarm"

    positive = finite[finite > 0]
    if positive.size and float(np.max(positive) / np.min(positive)) > 100:
        vmin = max(float(np.nanpercentile(positive, 1)), float(np.min(positive)))
        vmax = float(np.nanpercentile(positive, 99.5))
        if vmax > vmin:
            return LogNorm(vmin=vmin, vmax=vmax), "viridis"
    lo, hi = np.nanpercentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite) + 1.0)
    return Normalize(lo, hi), "viridis"


def plot_field(
    ax,
    polygons: np.ndarray,
    values: np.ndarray,
    title: str,
    units: str,
    *,
    signed: bool = False,
):
    """Plot one interior field on the physical R-Z cell polygons."""
    norm, cmap = _norm_and_cmap(values, signed=signed)
    collection = PolyCollection(
        polygons,
        array=np.asarray(values).ravel(),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=True,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    colorbar = ax.figure.colorbar(collection, ax=ax, pad=0.02, shrink=0.86)
    colorbar.set_label(units)
    return collection


def plot_gallery(
    parsed: Mapping,
    geometry: Mapping,
    items: Sequence[tuple[str, int]],
    *,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.2, 4.2),
):
    """Plot selected ``(field, component)`` pairs on the physical mesh."""
    polygons = cell_polygons(geometry)
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    spec_by_name = {spec.name: spec for spec in parsed["schema"]}
    for ax, (name, component) in zip(axes.flat, items):
        values, units = display_plane(parsed, name, component)
        suffix = f"[{component}]" if spec_by_name[name].components > 1 else ""
        plot_field(
            ax,
            polygons,
            values,
            f"{name}{suffix}: {spec_by_name[name].description}",
            units,
            signed=spec_by_name[name].signed,
        )
    for ax in axes.flat[len(items) :]:
        ax.set_visible(False)
    return fig, axes


def schema_table(parsed: Mapping) -> pd.DataFrame:
    """Return one row per logical field in file order."""
    return pd.DataFrame(
        [
            {
                "field": spec.name,
                "components": spec.components,
                "units": spec.units,
                "description": spec.description,
                "writer_placeholder": spec.writer_placeholder,
            }
            for spec in parsed["schema"]
        ]
    )
