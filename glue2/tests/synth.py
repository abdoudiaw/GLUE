"""Small synthetic events with the schema-v2 layout, for tests.

Each case has three controls (density, temperature, puff). Its EIRENE return
is a smooth function of the background, multiplied by Monte Carlo noise that
differs between repeated calls; the last stratum slot is unused (zero), as in
the real EIRBRA allocation.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

NX, NY, NSTRAT, NACTIVE = 6, 4, 3, 2
EV = 1.602176634e-19


def case_controls(index: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(1000 + index)
    return np.array([rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)]) * scale


def background(ctrl: np.ndarray) -> dict[str, np.ndarray]:
    dens, temp, puff = ctrl
    x = np.linspace(0, 1, NX)[None, :]
    y = np.linspace(0, 1, NY)[:, None]
    te = temp * 20 * EV * (1.2 - y) * (0.5 + x * (1 - x) * 2)
    ne = dens * 1e19 * (1 + 3 * y) * (1 + x)
    return {
        "braeir_dni": ne[None],
        "braeir_te": te,
        "braeir_ti": 1.1 * te,
        "braeir_vol": 1e-3 * (1 + 0 * x * y),
        "braeir_uu": (temp * 1e4 * (x - 0.5) * (1 + y))[None],
        "b2_tflux": np.array([puff * 1e21, dens * 3e20]),
        "b2_flux_scale": np.array([1.0, 0.5]),
    }


def response(bg: dict, rng: np.random.Generator, noise: float) -> dict[str, np.ndarray]:
    ne, te = bg["braeir_dni"][0], bg["braeir_te"]
    tev = te / EV
    rate = ne * np.sqrt(tev) * np.exp(-3.0 / tev)
    sni = np.zeros((NSTRAT, 1, NY, NX))
    for s, flux in enumerate(bg["b2_tflux"]):
        sni[s, 0] = flux * 1e-40 * rate * (1 + rng.normal(0, noise, rate.shape))
    see = -13.6 * EV * sni[:, 0] * (1 + 0.1 * tev / 10)
    sei = 0.3 * see
    smo = np.stack([sni[:, 0] * 1e-22 * bg["braeir_uu"][0], -0.2 * sni[:, 0] * 1e-22, 0 * sni[:, 0]], axis=1)
    dab2 = 1e40 * np.abs(sni[:, 0]).sum(0) / np.maximum(rate, 1e-300) * 1e-21
    return {"eirbra_sni": sni, "eirbra_see": see, "eirbra_sei": sei, "eirbra_smo": smo,
            "wneutrals_dab2": dab2[None, None]}


DIMS = {
    "braeir_dni": ("bra_fluid_species", "bra_y", "bra_x"),
    "braeir_uu": ("bra_fluid_species", "bra_y", "bra_x"),
    "braeir_te": ("bra_y", "bra_x"), "braeir_ti": ("bra_y", "bra_x"), "braeir_vol": ("bra_y", "bra_x"),
    "b2_tflux": ("active_stratum",), "b2_flux_scale": ("active_stratum",),
    "eirbra_sni": ("stratum_slot", "fluid_species", "eir_y", "eir_x"),
    "eirbra_see": ("stratum_slot", "eir_y", "eir_x"), "eirbra_sei": ("stratum_slot", "eir_y", "eir_x"),
    "eirbra_smo": ("stratum_slot", "momentum_slot", "eir_y", "eir_x"),
    "wneutrals_dab2": ("wneutral_state_slot", "wneutral_atom_species", "wneutral_y", "wneutral_x"),
}
SIZES = {"bra_fluid_species": 1, "bra_y": NY, "bra_x": NX, "active_stratum": NACTIVE,
         "stratum_slot": NSTRAT, "fluid_species": 1, "eir_y": NY, "eir_x": NX, "momentum_slot": 3,
         "wneutral_state_slot": 1, "wneutral_atom_species": 1, "wneutral_y": NY, "wneutral_x": NX}


def write_event(path: Path, arrays: dict, *, call: int, kind: str, rep: int, reps: int,
                schema: str = "2.0.0", used: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        for dim, size in SIZES.items():
            ds.createDimension(dim, size)
        for name, arr in arrays.items():
            ds.createVariable(name, "f8", DIMS[name])[:] = arr
        ds.createDimension("text1", 1)
        crc = ds.createVariable("b2_crcstra", "S1", ("active_stratum",))
        crc[:] = np.array([b"W", b"V"])
        ds.setncatts({
            "schema_name": "solps_eirene_training_event", "schema_version": schema,
            "event_kind": kind, "created_local": "20261002T000000.000-0400",
            "b2_5_git": "f03d24a", "solps_iter_git": "d6de341", "eirene_git": "f8f63fa",
            "b2_call_index": call, "eirene_repeat_index": rep, "eirene_repeat_count": reps,
        })
        if used is not None:
            ds.setncattr("eirene_result_used_by_b2", used)
    return path


def write_case(root: Path, index: int, *, repeats: int = 2, calls: int = 1, noise: float = 0.03,
               average: bool = True, scale: float = 1.0, schema: str = "2.0.0") -> list[Path]:
    """Write one run_<id>__D directory; returns the event paths.

    Schema-3 events also record which result B2 used: the average when one is
    written, otherwise the last repeat.
    """
    major = schema.split(".")[0]
    flagged = major != "2"
    averaged = average and repeats > 1
    case_dir = Path(root) / f"run_{index:08x}__D"
    paths = []
    for call in range(calls):
        ctrl = case_controls(index, scale) * (1 + 0.05 * call)
        bg = background(ctrl)
        rng = np.random.default_rng(index * 100 + call)
        outs = [response(bg, rng, noise) for _ in range(repeats)]
        for rep, out in enumerate(outs, start=1):
            name = f"eirene_training_v{major}_b2call_{call:08d}_single_call_{rep:04d}.nc"
            used = int(rep == repeats and not averaged) if flagged else None
            paths.append(write_event(case_dir / name, bg | out, call=call, kind="single_call",
                                     rep=rep, reps=repeats, schema=schema, used=used))
        if averaged:
            avg = {k: np.mean([o[k] for o in outs], axis=0) for k in outs[0]}
            name = f"eirene_training_v{major}_b2call_{call:08d}_average_used_by_b2_0000.nc"
            paths.append(write_event(case_dir / name, bg | avg, call=call, kind="average_used_by_b2",
                                     rep=0, reps=repeats, schema=schema, used=1 if flagged else None))
    return paths


INPUTS = ("braeir_dni", "braeir_te", "braeir_ti", "braeir_uu", "b2_tflux", "b2_flux_scale")
TARGETS = ("eirbra_sni", "eirbra_see", "eirbra_sei", "eirbra_smo", "wneutrals_dab2")


def mark_campaign_case(case_dir: Path, index: int, success: bool = True) -> None:
    """Add the sidecars the cloud campaign writes: sha256 manifest, controls, success marker."""
    import hashlib
    import json
    case_dir = Path(case_dir)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
             for p in sorted(case_dir.glob("eirene_training_v2_*.nc"))]
    (case_dir / "eirene_training_v2.sha256").write_text("\n".join(lines) + "\n")
    dens, temp, puff = case_controls(index)
    (case_dir / "source_params.json").write_text(json.dumps({"inputs": {
        "core": {"density_m-3": dens * 1e20}, "power": {"Pe_W": temp * 1e6},
        "gas_puffing": {"targets": {"D2": {"value": puff * 1e21, "gpfc": [2, 0, 0]}}}}}))
    if success:
        (case_dir / "EIRENE_TRAINING_SUCCESS").touch()
