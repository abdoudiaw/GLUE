#!/usr/bin/env python3
"""Perturbed restart states around a reference plasma state (numpy only, runs on Mora).

    python3 perturb_b2fstate.py REF_B2FSTATE OUT_B2FSTATI --seed 3 [--amplitude 0.2] [--modes 3]

Scales the ion densities (all fluid species, and ne with them), te and ti of
the reference state by smooth random fields exp(a * f(ix, iy)), where f is a
sum of low-order Fourier modes with unit RMS plus a random constant, so that
a is the typical log-amplitude of the perturbation (0.2 = about +-20 %).
Velocities, potential and fluxes are kept; B2.5 re-imposes its boundary
conditions on the first step. Every other block and the file layout are
preserved (same block structure as SOLSTICE's to_solps writer). A JSON
side-car records the seed and the realised min/max factors.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_dims(lines):
    for i, line in enumerate(lines):
        tok = line.split()
        if len(tok) >= 4 and tok[0] == "*cf:" and "nx,ny" in tok[3]:
            v = lines[i + 1].split()
            return int(v[0]), int(v[1]), int(v[2])
    raise RuntimeError("no nx,ny header")


def find_block(lines, name):
    for i, line in enumerate(lines):
        tok = line.split()
        if len(tok) >= 4 and tok[0] == "*cf:" and tok[3] == name:
            n, j, c = int(tok[2]), i + 1, 0
            while c < n:
                c += len(lines[j].split())
                j += 1
            return i + 1, j, n
    raise KeyError(name)


def get_block(lines, name):
    s, e, n = find_block(lines, name)
    return np.array(" ".join(lines[s:e]).split()[:n], dtype=np.float64)


def fmt(v, prec=17, width=26):
    m, ex = f"{v:.{prec}E}".split("E")
    return f"{m}E{ex[0]}{ex[1:].zfill(3)}".rjust(width)


def set_block(lines, name, flat, per_line=5):
    s, e, n = find_block(lines, name)
    assert len(flat) == n, (name, len(flat), n)
    lines[s:e] = ["".join(fmt(v) for v in flat[k:k + per_line]) + "\n" for k in range(0, n, per_line)]


def smooth_field(rng, nx, ny, modes):
    """Unit-RMS smooth random field on the (nx+2, ny+2) plane, Fortran order (ix fastest)."""
    x = np.linspace(0, 1, nx + 2)[:, None]
    y = np.linspace(0, 1, ny + 2)[None, :]
    f = np.zeros((nx + 2, ny + 2))
    for kx in range(modes + 1):
        for ky in range(modes + 1):
            if kx == ky == 0:
                continue
            a, b = rng.normal(size=2) / (1 + kx + ky)
            f += a * np.cos(np.pi * kx * x) * np.cos(np.pi * ky * y) + b * np.sin(np.pi * kx * x) * np.sin(np.pi * ky * y)
    f /= np.sqrt((f ** 2).mean()) + 1e-300
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--amplitude", type=float, default=0.2, help="log-amplitude of the smooth factor (0.2 ~ +-20 %%)")
    ap.add_argument("--constant", type=float, default=0.1, help="std of the extra global log-factor per field")
    ap.add_argument("--modes", type=int, default=3)
    a = ap.parse_args()
    lines = Path(a.ref).read_text().splitlines(keepends=True)
    nx, ny, ns = read_dims(lines)
    rng = np.random.default_rng(a.seed)
    ncell = (nx + 2) * (ny + 2)
    factors = {}
    for name in ("na", "te", "ti"):
        f = smooth_field(rng, nx, ny, a.modes).flatten(order="F")
        g = np.exp(a.amplitude * f + rng.normal(0, a.constant))
        factors[name] = g
    na = get_block(lines, "na").reshape(ncell, ns, order="F")
    na_new = na * factors["na"][:, None]
    set_block(lines, "na", na_new.flatten(order="F"))
    zamin = get_block(lines, "zamin")          # charge state per species -> ne = sum Z na (quasineutral)
    ne_new = (na_new * zamin[None, :]).sum(1)
    ne_old = get_block(lines, "ne")
    ne_ref = (na * zamin[None, :]).sum(1)
    # keep whatever offset the reference has between ne and sum Z na (should be ~0)
    set_block(lines, "ne", ne_new + (ne_old - ne_ref))
    for name in ("te", "ti"):
        set_block(lines, name, get_block(lines, name) * factors[name])
    Path(a.out).write_text("".join(lines))
    meta = {"ref": str(a.ref), "seed": a.seed, "amplitude": a.amplitude, "constant": a.constant, "modes": a.modes,
            "factor_range": {k: [float(v.min()), float(v.max())] for k, v in factors.items()},
            "ne_ref_consistency_max_rel": float(np.abs(ne_old - ne_ref).max() / np.abs(ne_old).max())}
    Path(str(a.out) + ".perturbation.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
