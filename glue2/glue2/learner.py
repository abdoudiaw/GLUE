"""Learner/bundle contract between GLUE and model libraries such as SOLSTICE.

A Learner trains on a frozen Snapshot and writes an immutable bundle directory
containing `bundle.json` (with at least `bundle_id`, `learner`, `snapshot_id`,
`inputs`, `targets`, `shapes`). A Bundle predicts the EIRENE return for raw
BRAEIR inputs and reports, per sample:

* mean     - {target: (n, *shape)} in the event file's native units;
* errbar   - {target: (n,)} calibrated error estimate in the learner's metric;
* score    - (n,) max over targets of errbar / held-out RMSE (higher = less trusted);
* novelty  - (n,) input distance from the training set (>1: outside it);
* ok       - (n,) True when the prediction may replace an EIRENE call.

This is the GLUE iserrok protocol extended to array-valued mesh fields.
Learners register by name; GLUE never imports a model library directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from glue2.snapshot import Snapshot


@dataclass
class Prediction:
    mean: dict[str, np.ndarray]
    errbar: dict[str, np.ndarray]
    score: np.ndarray
    novelty: np.ndarray
    ok: np.ndarray


class Bundle(Protocol):
    bundle_id: str
    meta: dict

    def predict(self, inputs: dict[str, np.ndarray]) -> Prediction: ...


class Learner(Protocol):
    name: str

    def fit(self, snapshot: Snapshot, out_root: str | Path) -> Path: ...


LEARNERS: dict[str, Callable[..., Learner]] = {}
BUNDLE_LOADERS: dict[str, Callable[[Path], Bundle]] = {}


def register(name: str, loader: Callable[[Path], Bundle]):
    def deco(factory):
        LEARNERS[name] = factory
        BUNDLE_LOADERS[name] = loader
        return factory
    return deco


def make_learner(name: str, **params) -> Learner:
    if name not in LEARNERS:
        raise KeyError(f"unknown learner {name!r}; registered: {sorted(LEARNERS)}")
    return LEARNERS[name](**params)


def load_bundle(path: str | Path) -> Bundle:
    path = Path(path)
    meta = json.loads((path / "bundle.json").read_text())
    return BUNDLE_LOADERS[meta["learner"]](path)


def evaluate(bundle: Bundle, snapshot: Snapshot, split: str = "test") -> dict:
    """Learner-agnostic metrics on one snapshot split, in native units.

    rel_l2: per-sample ||pred - true|| / ||true||, median over samples.
    sum_rel: relative error of the domain sum (e.g. total particle source),
    median over samples. gate_rate: fraction of samples the gate would accept.
    """
    data = snapshot.load(bundle.meta["inputs"] + bundle.meta["targets"], split)
    n = next(iter(data.values())).shape[0]
    if n == 0:
        return {"split": split, "n": 0}
    pred = bundle.predict({k: data[k] for k in bundle.meta["inputs"]})
    per_target = {}
    for name in bundle.meta["targets"]:
        truth = data[name].reshape(n, -1).astype(np.float64)
        guess = pred.mean[name].reshape(n, -1)
        norm = np.linalg.norm(truth, axis=1)
        rel = np.linalg.norm(guess - truth, axis=1) / np.where(norm > 0, norm, 1.0)
        tsum, gsum = truth.sum(1), guess.sum(1)
        sum_rel = np.abs(gsum - tsum) / np.where(np.abs(tsum) > 0, np.abs(tsum), 1.0)
        per_target[name] = {"rel_l2": float(np.median(rel)), "sum_rel": float(np.median(sum_rel))}
    return {
        "split": split,
        "n": int(n),
        "rel_l2": float(np.mean([v["rel_l2"] for v in per_target.values()])),
        "gate_rate": float(pred.ok.mean()),
        "per_target": per_target,
    }
