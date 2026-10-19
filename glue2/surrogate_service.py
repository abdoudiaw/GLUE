"""EIRENE surrogate service for B2.5's surrogate mode (eirene_training_replay = 3).

B2.5 writes, before each EIRENE call, the plasma background as the event file
`eirene_surrogate_request.nc` in its run directory (b2mn.exe.dir) and waits.
This service watches that directory, scores the background with a GLUE2
bundle and either

* writes `eirene_surrogate_answer.nc` (the request with the predicted
  per-stratum sources, and the wall tallies when the bundle has them, in the
  event layout: B2.5 applies it like a reduced replay and skips EIRENE), or
* creates the empty marker `eirene_surrogate_fallback` (B2.5 calls EIRENE; with
  the dump on, that call becomes a training event in the same directory).

The gate is the bundle's `ok` (errbar/rmse score and novelty), optionally
forced: `--always` answers every request (pure surrogate run), `--never` only
logs (shadow mode, EIRENE runs every step). `--gate-kinds sni,smo,see,sei`
restricts the score to those kinds (e.g. while the tally error bars are not
calibrated for a regime), `--skip-targets 'wneutrals_*'` leaves those
variables of the answer at the request's values (the previous EIRENE call's),
`--keep-requests DIR` keeps a copy of the requests, i.e. the plasma
background B2.5 sent at each call, for trajectory comparisons (every
`--keep-every` N-th call, default 1), and
`--max-consecutive N` forces an EIRENE call after N consecutive surrogate
answers (an anchor against drift: the gate judges single calls and cannot
see accumulated error). Decisions go to a JSONL log.
Answers are written to a temporary name and renamed, so B2.5 never reads a
partial file. Variables the bundle does not predict keep the request's values,
i.e. the previous EIRENE call's (today: the wall tallies, until the tally head
is trained).

    python -m glue2.surrogate_service --bundle <dir> --run-dir <b2mn.exe.dir> [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np

from glue2.events import read_arrays
from glue2.learner import Prediction, load_bundle

REQUEST = "eirene_surrogate_request.nc"
ANSWER = "eirene_surrogate_answer.nc"
FALLBACK = "eirene_surrogate_fallback"


def _rmse(bundle, name: str) -> float:
    """Held-out RMSE of a target, keyed by kind (SOLSTICE bundles) or by variable (baseline)."""
    kind = name.split("_", 1)[1]
    return float(bundle.rmse.get(kind, bundle.rmse.get(name, 1.0)))


def gate_score(bundle, pred, kinds: list[str] | None):
    """(score, ok) of the bundle, or restricted to the given kinds (errbar/rmse max over them)."""
    if not kinds:
        return float(pred.score[0]), bool(pred.ok[0])
    ratios = []
    for name, eb in pred.errbar.items():
        kind = name.split("_", 1)[1]
        if kind in kinds:
            ratios.append(float(eb[0]) / max(_rmse(bundle, name), 1e-12))
    if not ratios:
        raise ValueError(f"gate kinds {kinds} match none of {sorted(pred.errbar)}")
    score = max(ratios)
    return score, score <= bundle.fussiness and float(pred.novelty[0]) <= bundle.novelty_max


def write_answer(bundle, request: Path, answer: Path, pred, skip: list[str] | None = None,
                 keep_strata: list[str] | None = None, keep_momentum_slots: list[int] | None = None) -> dict:
    """Request file + prediction -> answer file (temporary name, then renamed).
    Variables matching a `skip` pattern keep the request's values. `keep_strata` (stratum type codes such as
    6C, 7V) and `keep_momentum_slots` (1 radial, 2 diamagnetic) keep those slices of the source arrays from
    the request, i.e. from the last EIRENE call: B2.5 rescales the puff and volume-recombination strata by
    the actual puff and recombination rates, so they matter more than their nominal flux suggests, and the
    model learns them with the floor weight."""
    import fnmatch
    tmp = answer.with_suffix(".nc.tmp")
    shutil.copyfile(request, tmp)
    summary = {}
    with netCDF4.Dataset(tmp, "a") as ds:
        for name in bundle.meta["targets"]:
            if any(fnmatch.fnmatch(name, pat) for pat in (skip or [])):
                summary[name] = "kept from request"
                continue
            var = ds.variables[name]
            var.set_auto_mask(False)
            new = np.asarray(pred.mean[name][0], dtype=np.float64)
            if new.shape != var.shape:
                raise ValueError(f"{name}: bundle shape {new.shape} vs request {var.shape}")
            if name.startswith("eirbra_") and (keep_strata or keep_momentum_slots):
                old = np.asarray(var[...], dtype=np.float64)
                for code in keep_strata or []:
                    k = bundle.keep[bundle.strata.index(code)]          # stratum slot of that type code
                    new[k] = old[k]
                if new.ndim == 4 and new.shape[1] > 1:                  # (slot, momentum slot, y, x)
                    for m in keep_momentum_slots or []:
                        new[:, m] = old[:, m]
            var[...] = new
            summary[name] = float(np.abs(new).sum())
        ds.setncattr("glue2_surrogate", json.dumps({"bundle_id": bundle.bundle_id, "score": float(pred.score[0]),
                                                    "novelty": float(pred.novelty[0])}))
    os.replace(tmp, answer)
    return summary


def serve(bundle_dir: str, run_dir: str, device: str | None, policy: str, log_path: str | None,
          poll: float = 0.05, once: bool = False, reload_file: str | None = None,
          gate_kinds: list[str] | None = None, skip_targets: list[str] | None = None,
          keep_requests: str | None = None, max_consecutive: int = 0, keep_every: int = 1,
          keep_strata: list[str] | None = None, keep_momentum_slots: list[int] | None = None,
          oracle_events: str | None = None, debias: str | None = None, oracle_window: int = 0) -> int:
    """oracle_events: directory of dumped EIRENE events of the same trajectory; the answer to call N is then the
    recorded EIRENE output of call N (every target, tallies included) instead of the model's prediction. A
    diagnostic of the coupling path itself: with the oracle the run must reproduce the EIRENE-driven run."""
    run = Path(run_dir)
    streak = 0
    keep = Path(keep_requests) if keep_requests else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(bundle_dir)
    if device and hasattr(bundle, "device"):
        bundle.device = device
    bias = dict(np.load(debias)) if debias else None     # diagnostic: mean (prediction - EIRENE) per target, event layout
    log = open(log_path or run / "eirene_surrogate_service.jsonl", "a")
    print(f"surrogate service: bundle {bundle.bundle_id}, watching {run}, policy {policy}", flush=True)
    req, ans, fb = run / REQUEST, run / ANSWER, run / FALLBACK
    n = 0
    while True:
        if reload_file and Path(reload_file).exists():          # hot swap of the bundle (online retrain)
            new_dir = Path(reload_file).read_text().strip()
            if new_dir and new_dir != str(bundle.path):
                try:
                    bundle = load_bundle(new_dir)
                    if device and hasattr(bundle, "device"):
                        bundle.device = device
                    print(f"surrogate service: switched to bundle {bundle.bundle_id}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"surrogate service: cannot load {new_dir}: {exc}", flush=True)
            Path(reload_file).unlink()
        if not req.exists():
            time.sleep(poll)
            continue
        t0 = time.time()
        try:
            with netCDF4.Dataset(req) as ds:
                call = int(ds.getncattr("b2_call_index"))
            inputs = {k: v[None] for k, v in read_arrays(req, bundle.meta["inputs"]).items()}
            pred = bundle.predict(inputs)
            if bias:
                for k, v in bias.items():
                    if k in pred.mean:
                        pred.mean[k] = pred.mean[k] - v[None]
            score, ok = gate_score(bundle, pred, gate_kinds)
            use = ok if policy == "gate" else policy == "always"
            anchor = bool(max_consecutive and use and streak >= max_consecutive)
            if anchor:
                use = False
            streak = streak + 1 if use else 0
            rec = {"call": call, "score": score, "novelty": float(pred.novelty[0]), "ok": ok, "anchor": anchor,
                   "score_all_kinds": float(pred.score[0]), "surrogate": use, "bundle": bundle.bundle_id,
                   "ratios": {n.split("_", 1)[1]: round(float(e[0]) / max(_rmse(bundle, n), 1e-12), 3)
                              for n, e in pred.errbar.items()}}
            if keep and call % max(int(keep_every), 1) == 0:
                shutil.copyfile(req, keep / f"request_call{call:06d}.nc")
            if use and oracle_events:
                import glob as _glob
                odir = oracle_events
                if "{seg}" in odir:                      # per-segment reference events: the run's current segment tag
                    tagf = Path(run_dir).resolve().parent.parent / "SEGTAG"
                    odir = odir.replace("{seg}", tagf.read_text().strip() if tagf.exists() else "seg_unknown")
                hits = sorted(_glob.glob(os.path.join(odir, f"*b2call_{call:08d}_*.nc")))
                if not hits:
                    raise FileNotFoundError(f"oracle: no event for call {call} in {odir}")
                truth = read_arrays(hits[0], bundle.meta["targets"])
                if oracle_window:                      # smooth oracle: running mean over neighbouring calls (unbiased, noise-free)
                    neigh = [sorted(_glob.glob(os.path.join(oracle_events, f"*b2call_{c:08d}_*.nc"))) for c in range(call - oracle_window, call + oracle_window + 1)]
                    neigh = [h[0] for h in neigh if h]
                    acc = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in truth.items()}
                    for h in neigh:
                        for k, v in read_arrays(h, bundle.meta["targets"]).items():
                            acc[k] += np.asarray(v, dtype=np.float64)
                    truth = {k: v / len(neigh) for k, v in acc.items()}
                    rec["oracle_window_n"] = len(neigh)
                oracle = Prediction({k: np.asarray(v)[None] for k, v in truth.items()}, {}, np.zeros(1), np.zeros(1), np.ones(1, bool))
                rec["sums"] = write_answer(bundle, req, ans, oracle, skip_targets, keep_strata, keep_momentum_slots)
                rec["oracle"] = os.path.basename(hits[0])
            elif use:
                rec["sums"] = write_answer(bundle, req, ans, pred, skip_targets, keep_strata, keep_momentum_slots)
            else:
                fb.touch()
            req.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            rec = {"error": repr(exc), "surrogate": False}
            fb.touch()
            req.unlink(missing_ok=True)
        rec["seconds"] = round(time.time() - t0, 3)
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: v for k, v in rec.items() if k not in ("sums", "ratios")}), flush=True)
        n += 1
        if once:
            return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--run-dir", required=True, help="directory B2.5 runs in (b2mn.exe.dir)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--always", action="store_true", help="answer every request (pure surrogate)")
    ap.add_argument("--never", action="store_true", help="shadow mode: score only, always fall back")
    ap.add_argument("--log", default=None)
    ap.add_argument("--once", action="store_true", help="serve one request and exit (tests)")
    ap.add_argument("--reload-file", default=None,
                    help="path of a file whose content names a new bundle directory to switch to")
    ap.add_argument("--gate-kinds", default=None, help="comma-separated kinds the gate scores (default: all)")
    ap.add_argument("--skip-targets", default=None,
                    help="comma-separated fnmatch patterns of target variables to keep from the request")
    ap.add_argument("--keep-requests", default=None, help="directory in which every request is kept")
    ap.add_argument("--keep-every", type=int, default=1, help="keep every N-th request only (with --keep-requests)")
    ap.add_argument("--max-consecutive", type=int, default=0,
                    help="force an EIRENE call after this many consecutive surrogate answers (0: never)")
    ap.add_argument("--keep-strata", default=None,
                    help="comma-separated stratum type codes (e.g. 6C,7V) whose source slices keep the request's values")
    ap.add_argument("--oracle-events", default=None,
                    help="diagnostic: answer call N with the recorded EIRENE event of call N from this directory")
    ap.add_argument("--oracle-window", type=int, default=0, help="with --oracle-events: answer with the mean of calls N-w..N+w")
    ap.add_argument("--debias", default=None, help="diagnostic: .npz of mean (prediction - EIRENE) per target to subtract")
    ap.add_argument("--keep-momentum-slots", default=None,
                    help="comma-separated momentum slots (1 radial, 2 diamagnetic) kept from the request")
    a = ap.parse_args(argv)
    policy = "always" if a.always else "never" if a.never else "gate"
    return serve(a.bundle, a.run_dir, a.device, policy, a.log, once=a.once, reload_file=a.reload_file,
                 gate_kinds=a.gate_kinds.split(",") if a.gate_kinds else None,
                 skip_targets=a.skip_targets.split(",") if a.skip_targets else None,
                 keep_requests=a.keep_requests, max_consecutive=a.max_consecutive, keep_every=a.keep_every,
                 keep_strata=a.keep_strata.split(",") if a.keep_strata else None,
                 keep_momentum_slots=[int(x) for x in a.keep_momentum_slots.split(",")] if a.keep_momentum_slots else None,
                 oracle_events=a.oracle_events, debias=a.debias, oracle_window=a.oracle_window)


if __name__ == "__main__":
    sys.exit(main())
