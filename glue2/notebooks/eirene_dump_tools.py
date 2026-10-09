"""Utilities for exploring schema-3 EIRENE training events.

An event file (``eirene_training_v3_b2call_*_single_call_*.nc``) holds one
EIRENE call at the ``eirene_eirsrt`` seam: the B2.5 background handed to
EIRENE (``braeir_*``, ``b2_*``) and the raw, unscaled state EIRENE returned
(``eirbra_*``, ``wneutrals_*``, ``cestim_*``).

Cell fields are stored on the B2 mesh including guard cells, so interior cell
``(ix, iy)`` is element ``[iy + 1, ix + 1]`` of the trailing two axes.  The
EIRBRA and WNEUTRALS allocations carry two unused trailing columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, SymLogNorm
import numpy as np
import pandas as pd
import xarray as xr

from fort31_tools import ELEMENTARY_CHARGE, cell_polygons, load_balance_geometry


# One-hue ramp for magnitudes; two opposed hues around a neutral grey for sign.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "magnitude",
    ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "signed",
    ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f6cdc6", "#ec8a7c", "#d03b3b", "#7f1d1d"],
)
SERIES_USED = "#2a78d6"
SERIES_WARMUP = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

CELL_DIMS = ({"eir_y", "eir_x"}, {"wneutral_y", "wneutral_x"}, {"bra_y", "bra_x"})
STRATUM_DIMS = ("stratum_slot", "wneutral_stratum_slot", "inventory_stratum_slot")
SPECIES_DIMS = (
    "fluid_species", "bra_fluid_species", "atom_species", "molecule_species",
    "test_ion_species", "wneutral_atom_species", "wneutral_molecule_species",
    "wneutral_test_ion_species", "wneutral_emission_channel", "wneutral_state_slot",
    "cestim_atom_species", "cestim_molecule_species", "cestim_test_ion_species",
)

# Return variables that describe the EIRENE set-up rather than the call result.
CONFIGURATION = {
    "eirene_index_npoint", "wneutrals_isrftype", "wneutrals_wlarea",
    "wneutrals_wlabsrp", "wneutrals_eirtxt", "wneutrals_sarea_res",
    "wneutrals_nds_ind", "wneutrals_nds_typ", "wneutrals_nds_srf",
    "wneutrals_nds_start", "wneutrals_nds_end",
}
JOULE_TEMPERATURES = {"braeir_te", "braeir_ti"}


def open_event(path: str | Path) -> xr.Dataset:
    """Load one training event fully into memory."""
    with xr.open_dataset(path) as ds:
        return ds.load()


def load_mesh(balance_path: str | Path) -> dict:
    """Return the B2 mesh and interior cell polygons from ``balance.nc``."""
    geometry = load_balance_geometry(balance_path)
    geometry["polygons"] = cell_polygons(geometry)
    return geometry


def load_triangles(fort33: str | Path, fort34: str | Path) -> dict:
    """Read the EIRENE triangle mesh: node coordinates [m] and connectivity."""
    values = Path(fort33).read_text().split()
    n_nodes = int(values[0])
    coordinates = np.array(values[1 : 1 + 2 * n_nodes], dtype=float) * 1.0e-2
    rows = np.loadtxt(fort34, skiprows=1, dtype=int)
    return {
        "r": coordinates[:n_nodes],
        "z": coordinates[n_nodes:],
        "triangles": rows[:, 1:4] - 1,
    }


def stratum_labels(ds: xr.Dataset) -> list[str]:
    """Return labels such as ``1 W`` for the active strata."""
    codes = [code.decode() if isinstance(code, bytes) else str(code) for code in ds["b2_crcstra"].values]
    return [f"{index + 1} {code.strip()}" for index, code in enumerate(codes)]


def stratum_weights(ds: xr.Dataset) -> np.ndarray:
    """Per-stratum factor B2.5 applies to raw EIRBRA estimators."""
    return np.asarray(ds["b2_tflux"].values * ds["b2_flux_scale"].values, dtype=float)


def group_of(name: str) -> str:
    if name.startswith(("braeir_", "b2_")):
        return "input"
    return name.split("_", 1)[0]


def layout_of(variable: xr.DataArray) -> str:
    dims = set(variable.dims)
    if any(pair <= dims for pair in CELL_DIMS):
        return "B2 cell field"
    if "triangle_cell" in dims:
        return "triangle field"
    if "resolved_surface_element" in dims:
        return "resolved surface"
    if dims & {"wall_tally_surface", "wall_property_surface"}:
        return "wall surface"
    if dims & set(STRATUM_DIMS) or "active_stratum" in dims:
        return "per stratum"
    return "other"


def _is_numeric(values: np.ndarray) -> bool:
    return values.dtype.kind in "fiu"


def variable_table(used: xr.Dataset, warmup: xr.Dataset | None = None) -> pd.DataFrame:
    """Classify every variable of an event.

    ``role`` separates what the surrogate has to produce from what an
    aggregator can restore.  With ``warmup`` (the discarded first call, which
    received identical inputs) a return variable is ``stochastic`` when the
    two calls differ and ``deterministic`` when they agree exactly.
    """
    rows = []
    for name, variable in used.variables.items():
        values = np.asarray(variable.values)
        numeric = _is_numeric(values)
        group = group_of(name)
        all_zero = bool(numeric and not np.any(values))
        same = None
        noise = np.nan
        if warmup is not None and name in warmup.variables:
            other = np.asarray(warmup[name].values)
            same = bool(np.array_equal(values, other))
            if numeric and not same and group != "input":
                scale = np.abs(values).max()
                mask = np.abs(values) > 1.0e-3 * scale
                if mask.any():
                    noise = float(np.median(np.abs(other[mask] - values[mask]) / np.abs(values[mask])))
        if group == "input":
            role = "input"
        elif name in CONFIGURATION:
            role = "configuration"
        elif all_zero:
            role = "zero in this case"
        elif same is None:
            role = "return"
        elif same:
            role = "deterministic return"
        else:
            role = "stochastic return"
        rows.append({
            "variable": name,
            "group": group,
            "layout": layout_of(variable),
            "shape": "×".join(str(size) for size in values.shape),
            "values": int(values.size),
            "nonzero_%": 100.0 * np.count_nonzero(values) / values.size if numeric else np.nan,
            "min": float(values.min()) if numeric else np.nan,
            "max": float(values.max()) if numeric else np.nan,
            "role": role,
            "call_to_call_noise": noise,
            "description": str(variable.attrs.get("long_name", "")),
        })
    return pd.DataFrame(rows)


def role_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Count variables and stored values by role."""
    summary = table.groupby("role").agg(variables=("variable", "count"), values=("values", "sum"))
    order = ["input", "stochastic return", "deterministic return", "configuration", "zero in this case", "return"]
    return summary.reindex([role for role in order if role in summary.index])


def cell_plane(
    ds: xr.Dataset,
    name: str,
    *,
    stratum: int | None = None,
    species: int = 0,
    component: int = 0,
) -> tuple[np.ndarray, str]:
    """Return one interior ``(ny, nx)`` plane and a colour-bar label.

    For EIRBRA arrays ``stratum=None`` gives the stratum sum B2.5 forms,
    ``sum(raw * tflux * flux_scale)``; an integer selects one raw stratum
    (0-based).  For WNEUTRALS stratum arrays slot 0 is the EIRENE total.
    """
    variable = ds[name]
    label = "native"
    for dim in variable.dims:
        if dim in SPECIES_DIMS:
            variable = variable.isel({dim: species})
        elif dim == "momentum_slot":
            variable = variable.isel({dim: component})
    if "stratum_slot" in variable.dims:
        if stratum is None:
            weights = stratum_weights(ds)
            active = variable.isel(stratum_slot=slice(0, weights.size))
            variable = (active * xr.DataArray(weights, dims="stratum_slot")).sum("stratum_slot")
            label = "Σ strata raw × tflux × flux_scale"
        else:
            variable = variable.isel(stratum_slot=stratum)
            label = "raw estimator"
    elif "wneutral_stratum_slot" in variable.dims:
        variable = variable.isel(wneutral_stratum_slot=0 if stratum is None else stratum + 1)
    plane = np.asarray(variable.values, dtype=float)[1:-1, 1:97]
    if name in JOULE_TEMPERATURES:
        return plane / ELEMENTARY_CHARGE, "eV"
    return plane, label


def _norm_and_cmap(values: np.ndarray):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0 or not np.any(finite):
        return Normalize(0, 1), SEQUENTIAL
    if np.max(finite) <= 0 and float(np.min(finite) / np.max(finite[finite < 0])) <= 100:
        return Normalize(float(np.min(finite)), float(np.max(finite))), SEQUENTIAL.reversed()
    if np.min(finite) < 0:
        limit = float(np.nanpercentile(np.abs(finite), 99.5)) or float(np.abs(finite).max())
        magnitudes = np.abs(finite[np.nonzero(finite)])
        linthresh = max(float(np.nanpercentile(magnitudes, 10)), limit * 1.0e-5)
        return SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit, base=10), DIVERGING
    positive = finite[finite > 0]
    if positive.size and float(positive.max() / positive.min()) > 100:
        vmin = max(float(np.nanpercentile(positive, 1)), float(positive.max()) * 1.0e-6)
        vmax = float(np.nanpercentile(positive, 99.5))
        if vmax > vmin:
            return LogNorm(vmin=vmin, vmax=vmax), SEQUENTIAL
    lo, hi = np.nanpercentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max()) or 1.0
        if hi <= lo:
            hi = lo + 1.0
    return Normalize(lo, hi), SEQUENTIAL


def _style_colorbar(colorbar, norm, label: str):
    if isinstance(norm, SymLogNorm):
        low = int(np.ceil(np.log10(norm.linthresh)))
        high = int(np.floor(np.log10(norm.vmax)))
        step = max(1, int(np.ceil((high - low + 1) / 3)))
        decades = 10.0 ** np.arange(high, low - 1, -step)[::-1]
        if decades.size:
            colorbar.set_ticks(np.concatenate((-decades[::-1], [0.0], decades)))
            colorbar.minorticks_off()
    colorbar.set_label(label, fontsize=7, color=MUTED)
    colorbar.ax.tick_params(labelsize=7, colors=MUTED)
    colorbar.outline.set_visible(False)


def _style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_plane(ax, polygons: np.ndarray, values: np.ndarray, title: str, label: str, *, norm=None, cmap=None):
    """Draw one interior plane on the physical R-Z cell polygons."""
    if norm is None:
        norm, cmap = _norm_and_cmap(values)
    collection = PolyCollection(
        polygons, array=np.asarray(values).ravel(), cmap=cmap, norm=norm,
        edgecolors="none", rasterized=True,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel("R [m]", fontsize=8, color=MUTED)
    ax.set_ylabel("Z [m]", fontsize=8, color=MUTED)
    _style_axis(ax)
    colorbar = ax.figure.colorbar(collection, ax=ax, pad=0.02, shrink=0.86)
    _style_colorbar(colorbar, norm, label)
    return collection


def _grid(count: int, ncols: int, panel: tuple[float, float]):
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
        constrained_layout=True, squeeze=False,
    )
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, axes


def short_name(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def plot_cell_gallery(
    ds: xr.Dataset,
    mesh: Mapping,
    items: Sequence[str | tuple[str, dict]],
    *,
    ncols: int = 4,
    panel: tuple[float, float] = (3.6, 4.0),
):
    """Plot cell fields; each item is a name or ``(name, cell_plane kwargs)``."""
    items = [(item, {}) if isinstance(item, str) else item for item in items]
    fig, axes = _grid(len(items), ncols, panel)
    for ax, (name, options) in zip(axes.flat, items):
        values, label = cell_plane(ds, name, **options)
        suffix = "" if not options else " " + ", ".join(f"{key}={value}" for key, value in options.items())
        plot_plane(ax, mesh["polygons"], values, short_name(name) + suffix, label)
    return fig, axes


def cell_field_names(ds: xr.Dataset, prefix: str, *, skip_zero: bool = True) -> list[str]:
    """List the cell-field variables of one group, in file order."""
    names = []
    for name, variable in ds.variables.items():
        if not name.startswith(prefix) or layout_of(variable) != "B2 cell field":
            continue
        if skip_zero and not np.any(variable.values):
            continue
        names.append(name)
    return names


def plot_strata(
    ds: xr.Dataset,
    mesh: Mapping,
    name: str,
    *,
    species: int = 0,
    component: int = 0,
    panel: tuple[float, float] = (3.1, 3.6),
):
    """Small multiples of one raw EIRBRA estimator, one panel per stratum."""
    labels = stratum_labels(ds)
    fig, axes = _grid(len(labels), len(labels), panel)
    for index, (ax, label) in enumerate(zip(axes.flat, labels)):
        values, unit = cell_plane(ds, name, stratum=index, species=species, component=component)
        plot_plane(ax, mesh["polygons"], values, f"{short_name(name)} — stratum {label}", unit)
    return fig, axes


def plot_call_comparison(
    used: xr.Dataset,
    warmup: xr.Dataset,
    mesh: Mapping,
    items: Sequence[str | tuple[str, dict]],
    *,
    panel: tuple[float, float] = (3.6, 4.0),
):
    """Used call, warm-up call and their relative difference, one row per field."""
    items = [(item, {}) if isinstance(item, str) else item for item in items]
    fig, axes = plt.subplots(
        len(items), 3, figsize=(panel[0] * 3, panel[1] * len(items)),
        constrained_layout=True, squeeze=False,
    )
    for row, (name, options) in zip(axes, items):
        second, label = cell_plane(used, name, **options)
        first, _ = cell_plane(warmup, name, **options)
        norm, cmap = _norm_and_cmap(second)
        plot_plane(row[0], mesh["polygons"], second, f"{short_name(name)} — call 2 (used)", label, norm=norm, cmap=cmap)
        plot_plane(row[1], mesh["polygons"], first, f"{short_name(name)} — call 1 (warm-up)", label, norm=norm, cmap=cmap)
        relative = np.where(second != 0, (first - second) / np.where(second != 0, np.abs(second), 1.0), np.nan)
        plot_plane(
            row[2], mesh["polygons"], relative, f"{short_name(name)} — (call 1 − call 2) / |call 2|",
            "relative difference", norm=Normalize(-1, 1), cmap=DIVERGING,
        )
    return fig, axes


def plot_noise_ranking(table: pd.DataFrame, *, top: int = 40):
    """Median call-to-call relative difference of the stochastic return fields."""
    data = table.dropna(subset=["call_to_call_noise"]).sort_values("call_to_call_noise").tail(top)
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(data) + 1.0), constrained_layout=True)
    ax.barh(data["variable"], 100.0 * data["call_to_call_noise"], color=SERIES_USED, height=0.62)
    ax.set_xscale("log")
    ticks = [tick for tick in (1, 3, 10, 30, 100, 300, 1000) if tick <= 100.0 * data["call_to_call_noise"].max() * 3]
    ax.set_xticks(ticks, [str(tick) for tick in ticks])
    ax.minorticks_off()
    ax.set_xlabel("median |call 1 − call 2| / |call 2|  [%]", fontsize=9, color=MUTED)
    ax.set_title("Difference between two calls with identical inputs", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style_axis(ax)
    ax.tick_params(axis="y", labelsize=7)
    return fig, ax


def stratum_table(ds: xr.Dataset) -> pd.DataFrame:
    """Per-stratum controls and EIRBRA scalars, one row per active stratum."""
    count = ds.sizes["active_stratum"]
    frame = pd.DataFrame({
        "stratum": stratum_labels(ds),
        "tflux": ds["b2_tflux"].values,
        "flux_scale": ds["b2_flux_scale"].values,
    })
    for name in ("eirbra_volsumn", "eirbra_volsumee", "eirbra_volsumei", "eirbra_srcstrn"):
        frame[short_name(name)] = np.asarray(ds[name].values)[:count]
    frame["srccrfc"] = np.asarray(ds["eirbra_srccrfc"].values)[:count, 0]
    return frame


def surface_names(ds: xr.Dataset, layout: str, *, skip_zero: bool = True) -> list[str]:
    names = []
    for name, variable in ds.variables.items():
        if layout_of(variable) != layout or not _is_numeric(np.asarray(variable.values)):
            continue
        if name in CONFIGURATION and name not in {"wneutrals_wlarea", "wneutrals_sarea_res"}:
            continue
        if skip_zero and not np.any(variable.values):
            continue
        names.append(name)
    return names


def _surface_series(ds: xr.Dataset, name: str) -> np.ndarray:
    """Reduce a surface array to one value per surface: stratum total, first species."""
    variable = ds[name]
    for dim in variable.dims:
        if dim == "wneutral_stratum_slot":
            variable = variable.isel({dim: 0})
        elif dim in SPECIES_DIMS or dim in {"wall_plasma_species", "wall_particle_species"}:
            variable = variable.isel({dim: 0})
    return np.asarray(variable.values, dtype=float)


def plot_surfaces(
    used: xr.Dataset,
    names: Sequence[str],
    *,
    warmup: xr.Dataset | None = None,
    ncols: int = 4,
    panel: tuple[float, float] = (3.6, 2.3),
    xlabel: str = "surface index",
):
    """Surface tallies against surface index (stratum total, first species)."""
    fig, axes = _grid(len(names), ncols, panel)
    for ax, name in zip(axes.flat, names):
        series = _surface_series(used, name)
        index = np.arange(1, series.size + 1)
        if warmup is not None:
            ax.plot(index, _surface_series(warmup, name), color=SERIES_WARMUP, linewidth=1.2, label="call 1 (warm-up)")
        ax.plot(index, series, color=SERIES_USED, linewidth=1.6, label="call 2 (used)")
        positive = series[series > 0]
        if series.min() >= 0 and positive.size and positive.max() / positive.min() > 100:
            ax.set_yscale("log")
        ax.set_title(short_name(name), fontsize=9, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style_axis(ax)
    if warmup is not None and len(names):
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncols=2, frameon=False, fontsize=8)
    return fig, axes


def plot_triangles(
    ds: xr.Dataset,
    triangles: Mapping,
    names: Sequence[str],
    *,
    ncols: int = 3,
    panel: tuple[float, float] = (3.8, 4.4),
):
    """Triangle-grid estimators on the EIRENE mesh (first species)."""
    fig, axes = _grid(len(names), ncols, panel)
    count = len(triangles["triangles"])
    for ax, name in zip(axes.flat, names):
        values = np.asarray(ds[name].values, dtype=float)[:count, 0]
        norm, cmap = _norm_and_cmap(values)
        image = ax.tripcolor(
            triangles["r"], triangles["z"], triangles["triangles"], facecolors=values,
            cmap=cmap, norm=norm, rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(short_name(name), fontsize=9, color=INK)
        ax.set_xlabel("R [m]", fontsize=8, color=MUTED)
        ax.set_ylabel("Z [m]", fontsize=8, color=MUTED)
        _style_axis(ax)
        colorbar = fig.colorbar(image, ax=ax, pad=0.02, shrink=0.86)
        _style_colorbar(colorbar, norm, "native")
    return fig, axes
