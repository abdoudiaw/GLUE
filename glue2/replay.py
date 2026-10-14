"""Replay event files for the B2.5 surrogate test.

B2.5 (fork branch feature/eirene-training-dump) has a replay hook: with
``eirene_training_replay = 2`` and ``eirene_training_replay_file = <nc>`` it
replaces, after every EIRENE call, the four per-stratum sources and the six
wall tallies by the content of the named event file and zeroes the rest of
the return. This module writes such files from a GLUE2 bundle:

* ``surrogate``: a copy of a template event (the EIRENE call whose background
  the bundle predicts for) with every variable the bundle predicts replaced by
  its prediction, in the event layout (guard cells, unused strata and
  momentum slots the bundle does not predict are zero);
* ``masked_truth``: the same template with the bundle's structural zeros
  applied to the true arrays and nothing else changed. One B2.5 step on it
  tells whether the entries the bundle leaves at zero matter at all, so the
  surrogate step can be judged against it as well as against the true step.

Variables the bundle does not predict (today: the wall tallies) stay as in
the template, which isolates the source prediction.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np

from glue2.events import read_arrays
from glue2.learner import Bundle, Prediction, load_bundle


def predict_event(bundle: Bundle, event: str | Path) -> Prediction:
    """The bundle's prediction for the background of one event file."""
    return bundle.predict(_as_batch(read_arrays(event, bundle.meta["inputs"])))


def _as_batch(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v)[None] for k, v in arrays.items()}


def write_replay_event(bundle: Bundle, template: str | Path, out: str | Path,
                       mode: str = "surrogate", prediction: Prediction | None = None) -> dict:
    """Copy ``template`` to ``out`` with the bundle's targets replaced. Returns a
    summary {variable: {"rel_l2": ..., "zeroed_fraction": ...}} against the template."""
    template, out = Path(template), Path(out)
    if mode not in ("surrogate", "masked_truth"):
        raise ValueError("mode must be surrogate or masked_truth")
    if prediction is None:
        prediction = predict_event(bundle, template)
    truth = read_arrays(template, bundle.meta["targets"])
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, out)
    summary = {"mode": mode, "bundle_id": bundle.bundle_id, "template": str(template),
               "score": float(prediction.score[0]), "novelty": float(prediction.novelty[0]),
               "ok": bool(prediction.ok[0]), "variables": {}}
    with netCDF4.Dataset(out, "a") as ds:
        for name in bundle.meta["targets"]:
            pred = np.asarray(prediction.mean[name][0], dtype=np.float64)
            true = np.asarray(truth[name], dtype=np.float64)
            if pred.shape != true.shape:
                raise ValueError(f"{name}: bundle shape {pred.shape} vs event {true.shape}")
            keep = pred != 0
            new = pred if mode == "surrogate" else true * keep
            var = ds.variables[name]
            var.set_auto_mask(False)
            var[...] = new
            norm = np.linalg.norm(true)
            summary["variables"][name] = {
                "rel_l2": float(np.linalg.norm(new - true) / norm) if norm > 0 else 0.0,
                "zeroed_fraction": float(np.abs(true[~keep]).sum() / np.abs(true).sum()) if norm > 0 else 0.0,
            }
        ds.setncattr("glue2_replay", json.dumps({k: v for k, v in summary.items() if k != "variables"}))
    return summary


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="write B2.5 replay event files from a GLUE2 bundle")
    ap.add_argument("--bundle", required=True, help="bundle directory")
    ap.add_argument("--template", required=True, help="event file of the EIRENE call to replace")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda for SOLSTICE bundles")
    a = ap.parse_args(argv)
    bundle = load_bundle(a.bundle)
    if a.device and hasattr(bundle, "device"):
        bundle.device = a.device
    pred = predict_event(bundle, a.template)
    out = {}
    for mode in ("surrogate", "masked_truth"):
        path = Path(a.out_dir) / f"{mode}.nc"
        out[mode] = write_replay_event(bundle, a.template, path, mode, pred)
        print(f"{path}: score {out[mode]['score']:.3f} novelty {out[mode]['novelty']:.3f} ok {out[mode]['ok']}")
        for name, v in out[mode]["variables"].items():
            print(f"   {name:18s} rel_l2 vs event {v['rel_l2']:.3f}   mass in zeroed entries {v['zeroed_fraction']:.2e}")
    (Path(a.out_dir) / "replay_summary.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
