"""Helpers for the hero-run notebook: b2fstate reader, mesh polygons, concatenated histories."""
from __future__ import annotations

import glob
import re
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr
from matplotlib.collections import PolyCollection

NX, NY = 96, 36
EV = 1.602176634e-19
DEFAULT_RUN = Path("~/data/eirene_nn/reference_uniform_dt1e-5").expanduser()
DEFAULT_MESH = Path("~/data/eirene_nn/stores/diiid_appfpp_eirene_v3_mesh.nc").expanduser()


def read_b2fstate(path):
    """{field: array} of the real fields of a b2fstate/b2fstati text file."""
    fields, name, kind, buf = {}, None, None, []

    def flush():
        if name is not None and kind in ("real", "int"):
            fields[name] = np.array(buf, dtype=float if kind == "real" else int)

    with open(path) as f:
        for line in f:
            if line.startswith("*cf:"):
                flush()
                parts = line.split()
                kind, name, buf = parts[1], parts[3], []
            elif kind in ("real", "int"):
                buf.extend(line.split())
    flush()
    return fields


def grid(v, nx=NX, ny=NY):
    """b2fstate vector -> (component, iy, ix) over the mesh with guard cells (Fortran order, ix fastest)."""
    return v.reshape(-1, ny + 2, nx + 2)


def interior(a):
    return a[..., 1:-1, 1:-1]


def state_fields(path):
    """Interior ion density (D+), Te and Ti in eV, parallel velocity, potential as (iy, ix) arrays."""
    s = read_b2fstate(path)
    return {
        "n_D+ [m^-3]": interior(grid(s["na"])[1]),
        "Te [eV]": interior(grid(s["te"])[0]) / EV,
        "Ti [eV]": interior(grid(s["ti"])[0]) / EV,
        "u_par D+ [m/s]": interior(grid(s["ua"])[1]),
        "potential [V]": interior(grid(s["po"])[0]),
    }


def mesh_polygons(mesh_path=DEFAULT_MESH):
    """Cell polygons (R, Z) in (iy, ix) order of the interior mesh, from the SOLSTICE mesh store."""
    ds = xr.open_dataset(mesh_path)
    r = ds["cell_corners_r"].values
    z = ds["cell_corners_z"].values
    ix = ds["cell_ix"].values - 1
    iy = ds["cell_iy"].values - 1
    polys = np.empty((NY, NX), dtype=object)
    for k in range(len(ix)):
        polys[iy[k], ix[k]] = np.column_stack([r[k], z[k]])
    return polys


def plot_field(ax, polys, values, title="", cmap="viridis", log=False, vmin=None, vmax=None):
    """Colour the mesh cells by a (iy, ix) field."""
    import matplotlib.colors as mcolors
    verts = [polys[iy, ix] for iy in range(NY) for ix in range(NX)]
    vals = np.array([values[iy, ix] for iy in range(NY) for ix in range(NX)])
    norm = mcolors.LogNorm(vmin=vmin or max(vals[vals > 0].min(), 1e-300), vmax=vmax or vals.max()) if log else \
        mcolors.Normalize(vmin=vals.min() if vmin is None else vmin, vmax=vals.max() if vmax is None else vmax)
    pc = PolyCollection(verts, array=vals, cmap=cmap, norm=norm, edgecolors="none")
    ax.add_collection(pc)
    ax.set_xlim(min(v[:, 0].min() for v in verts), max(v[:, 0].max() for v in verts))
    ax.set_ylim(min(v[:, 1].min() for v in verts), max(v[:, 1].max() for v in verts))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return pc


def histories(run=DEFAULT_RUN, keys=("nesepm", "tesepm", "tisepm", "fnixap", "fnixip", "temxap", "temxip", "tflux")):
    """Concatenate the per-segment b2time.nc histories into one record with absolute time and step."""
    files = sorted(glob.glob(str(Path(run) / "states" / "ref_b2time_*.nc")))
    out = {k: [] for k in keys}
    out["time"], out["step"] = [], []
    t_off, s_off = 0.0, 50
    for f in files:
        ds = netCDF4.Dataset(f)
        t = ds.variables["timesa"][:]
        n = len(t)
        for k in keys:
            if k in ds.variables:
                v = ds.variables[k][:]
                out[k].append(v.reshape(n, -1) if v.ndim > 1 else v.reshape(n, 1))
        end_step = int(re.search(r"(\d{5})\.nc$", f).group(1))
        seg_steps = end_step - s_off
        out["time"].append(t_off + t)
        out["step"].append(s_off + np.round(t / t[-1] * seg_steps) if t[-1] > 0 else s_off + np.zeros(n))
        t_off += t[-1]
        s_off = end_step
    return {k: (np.concatenate(v) if v else None) for k, v in out.items()}


def segment_walltimes(run=DEFAULT_RUN):
    """[(end step, seconds)] from the chain log."""
    out = []
    for line in open(Path(run) / "reference_chain.log"):
        m = re.match(r"segment to step (\d+): (\d+) s", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)
