"""Baseline learner: signed-log transform, block-scaled PCA, bootstrap ridge ensemble.

It exists to exercise the GLUE loop end to end and to give the SOLSTICE EIRENE
model a reference number to beat, not to be deployed.

* Every variable is mapped z = asinh(x / s), s = median |x| over non-zero
  training entries: sources span tens of decades and change sign.
* Each variable block is divided by its RMS so all variables weigh equally,
  then inputs and targets are reduced by PCA.
* Ridge maps input scores (plus pairwise products of the leading scores for
  degree 2) to target scores (alpha by leave-one-out); members are refitted on bootstrap resamples of whole cases, and their spread is the
  raw error bar.
* Calibration on the validation split sets, per target, the error-bar scale and
  the held-out RMSE used by the ok gate (as in GLUE's iserrok).
* Novelty is the mean k-NN distance of the input scores to the training set,
  divided by the largest such leave-one-out distance among training samples.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np

from glue2.catalog import utcnow
from glue2.learner import Prediction, register
from glue2.snapshot import Snapshot

ALPHAS = np.logspace(-4, 3, 15)


def _scale(x: np.ndarray) -> float:
    nz = np.abs(x[x != 0])
    return float(np.median(nz)) if nz.size else 1.0


class _Blocks:
    """Per-variable asinh transform and block RMS scaling to one flat matrix."""

    def __init__(self, names, shapes, scales, centers, rms):
        self.names, self.shapes = list(names), {k: tuple(v) for k, v in shapes.items()}
        self.scales, self.centers, self.rms = scales, centers, rms

    @classmethod
    def fit(cls, data: dict[str, np.ndarray], names) -> "_Blocks":
        scales, centers, rms = {}, {}, {}
        for name in names:
            x = data[name].reshape(len(data[name]), -1).astype(np.float64)
            scales[name] = _scale(x)
            z = np.arcsinh(x / scales[name])
            centers[name] = z.mean(0)
            rms[name] = float(np.sqrt(((z - centers[name]) ** 2).mean())) or 1.0
        return cls(names, {k: data[k].shape[1:] for k in names}, scales, centers, rms)

    def forward(self, data: dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for name in self.names:
            x = np.asarray(data[name], dtype=np.float64).reshape(len(data[name]), -1)
            cols.append((np.arcsinh(x / self.scales[name]) - self.centers[name]) / self.rms[name])
        return np.concatenate(cols, axis=1)

    def slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for name in self.names:
            size = int(np.prod(self.shapes[name]))
            out[name] = slice(start, start + size)
            start += size
        return out

    def inverse(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for name, sl in self.slices().items():
            z = flat[:, sl] * self.rms[name] + self.centers[name]
            out[name] = (self.scales[name] * np.sinh(z)).reshape(len(flat), *self.shapes[name])
        return out

    def state(self, prefix: str) -> dict[str, np.ndarray]:
        st = {}
        for name in self.names:
            st[f"{prefix}/{name}/center"] = self.centers[name]
        return st

    def meta(self) -> dict:
        return {"names": self.names, "shapes": {k: list(v) for k, v in self.shapes.items()},
                "scales": self.scales, "rms": self.rms}

    @classmethod
    def restore(cls, meta: dict, arrays, prefix: str) -> "_Blocks":
        centers = {n: arrays[f"{prefix}/{n}/center"] for n in meta["names"]}
        return cls(meta["names"], meta["shapes"], meta["scales"], centers, meta["rms"])


def _pca(X: np.ndarray, max_k: int, var_keep: float) -> np.ndarray:
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    frac = np.cumsum(s**2) / max((s**2).sum(), 1e-300)
    k = int(min(max_k, np.searchsorted(frac, var_keep) + 1, len(s)))
    return vt[:max(k, 1)]


def _features(Z: np.ndarray, degree: int, poly_k: int) -> np.ndarray:
    """Scores plus, for degree 2, products of the leading poly_k scores."""
    if degree == 1:
        return Z
    lead = Z[:, :poly_k]
    iu = np.triu_indices(lead.shape[1])
    return np.concatenate([Z, (lead[:, :, None] * lead[:, None, :])[:, iu[0], iu[1]]], axis=1)


def _ridge(Z: np.ndarray, T: np.ndarray, alpha: float | None = None):
    """Ridge with intercept; alpha by leave-one-out when not given."""
    zm, tm = Z.mean(0), T.mean(0)
    Zc, Tc = Z - zm, T - tm
    u, s, vt = np.linalg.svd(Zc, full_matrices=False)
    if alpha is None:
        best = np.inf
        for a in ALPHAS:
            d = s**2 / (s**2 + a)
            H = (u * d) @ u.T
            resid = (Tc - H @ Tc) / np.clip(1 - np.diag(H) - 1 / len(Z), 1e-6, None)[:, None]
            loo = float((resid**2).mean())
            if loo < best:
                best, alpha = loo, float(a)
    W = vt.T @ np.diag(s / (s**2 + alpha)) @ u.T @ Tc
    return W, tm - zm @ W, alpha


class PCARidgeBundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "bundle.json").read_text())
        self.bundle_id = self.meta["bundle_id"]
        a = np.load(self.path / "arrays.npz")
        self.inp = _Blocks.restore(self.meta["blocks"]["inputs"], a, "in")
        self.out = _Blocks.restore(self.meta["blocks"]["targets"], a, "out")
        self.in_basis, self.out_basis = a["in_basis"], a["out_basis"]
        self.score_scale = a["score_scale"]
        self.W, self.b = a["W"], a["b"]
        self.train_scores = a["train_scores"]
        cal = self.meta["calibration"]
        self.errbar_scale = {k: float(v) for k, v in cal["errbar_scale"].items()}
        self.rmse = {k: float(v) for k, v in cal["rmse"].items()}
        self.knn_ref = float(cal["knn_ref"])
        self.fussiness = float(self.meta["params"]["fussiness"])
        self.novelty_max = float(self.meta["params"]["novelty_max"])
        self.k = int(self.meta["params"]["knn"])
        self.degree = int(self.meta["params"]["degree"])
        self.poly_k = int(self.meta["params"]["poly_components"])

    def scores(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return (self.inp.forward(inputs) @ self.in_basis.T) / self.score_scale

    def _raw(self, inputs):
        Z = self.scores(inputs)
        F = _features(Z, self.degree, self.poly_k)
        members = np.einsum("nk,mkj->mnj", F, self.W) + self.b[:, None, :]   # (member, n, out_k)
        flat = members @ self.out_basis                                    # (member, n, q)
        return Z, flat.mean(0), flat.std(0)

    def novelty(self, Z: np.ndarray) -> np.ndarray:
        d = np.sqrt(((Z[:, None, :] - self.train_scores[None]) ** 2).sum(-1))
        k = min(self.k, self.train_scores.shape[0])
        return np.sort(d, axis=1)[:, :k].mean(1) / self.knn_ref

    def predict(self, inputs: dict[str, np.ndarray]) -> Prediction:
        Z, mean_flat, std_flat = self._raw(inputs)
        n = len(Z)
        errbar, ratio = {}, np.zeros((n, len(self.out.names)))
        for j, (name, sl) in enumerate(self.out.slices().items()):
            errbar[name] = self.errbar_scale[name] * std_flat[:, sl].mean(1)
            ratio[:, j] = errbar[name] / max(self.rmse[name], 1e-12)
        novelty = self.novelty(Z)
        score = ratio.max(1)
        ok = (score <= self.fussiness) & (novelty <= self.novelty_max)
        return Prediction(self.out.inverse(mean_flat), errbar, score, novelty, ok)


@register("pca_ridge", PCARidgeBundle)
class PCARidgeLearner:
    name = "pca_ridge"

    def __init__(self, members: int = 8, max_components: int = 32, var_keep: float = 0.999,
                 fussiness: float = 1.5, novelty_max: float = 1.0, knn: int = 5, seed: int = 0,
                 degree: int = 2, poly_components: int = 6):
        self.params = dict(members=members, max_components=max_components, var_keep=var_keep,
                           fussiness=fussiness, novelty_max=novelty_max, knn=knn, seed=seed,
                           degree=degree, poly_components=poly_components)

    def fit(self, snapshot: Snapshot, out_root: str | Path) -> Path:
        p = self.params
        spec = snapshot.spec
        names = list(spec.inputs) + list(spec.targets)
        train = snapshot.load(names, "train")
        n = len(train[spec.inputs[0]])
        if n < 3:
            raise ValueError(f"need at least 3 training samples, have {n}")
        cases = np.array([e["case_id"] for e in snapshot.events("train")])

        inp = _Blocks.fit(train, spec.inputs)
        out = _Blocks.fit(train, spec.targets)
        X, Y = inp.forward(train), out.forward(train)
        in_basis = _pca(X, min(p["max_components"], n - 1), p["var_keep"])
        out_basis = _pca(Y, min(p["max_components"], n - 1), p["var_keep"])
        raw_scores = X @ in_basis.T
        score_scale = raw_scores.std(0) + 1e-12
        Z, T = raw_scores / score_scale, Y @ out_basis.T

        F = _features(Z, p["degree"], p["poly_components"])
        _, _, alpha = _ridge(F, T)
        rng = np.random.default_rng(p["seed"])
        uniq = np.unique(cases)
        Ws, bs = [], []
        for m in range(p["members"]):
            if m == 0:
                rows = np.arange(n)
            else:
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                rows = np.concatenate([np.flatnonzero(cases == c) for c in pick])
            W, b, _ = _ridge(F[rows], T[rows], alpha)
            Ws.append(W)
            bs.append(b)
        W, b = np.stack(Ws), np.stack(bs)

        d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        k = min(p["knn"], n - 1)
        knn_ref = float(np.sort(d, axis=1)[:, :k].mean(1).max())

        arrays = {"in_basis": in_basis, "out_basis": out_basis, "score_scale": score_scale,
                  "W": W, "b": b, "train_scores": Z, **inp.state("in"), **out.state("out")}
        meta = {
            "learner": self.name,
            "snapshot_id": snapshot.snapshot_id,
            "inputs": list(spec.inputs),
            "targets": list(spec.targets),
            "shapes": {k: list(v) for k, v in snapshot.shapes.items()},
            "params": p,
            "alpha": alpha,
            "n_train": n,
            "components": {"inputs": int(len(in_basis)), "targets": int(len(out_basis))},
            "blocks": {"inputs": inp.meta(), "targets": out.meta()},
            "calibration": {"knn_ref": knn_ref, "errbar_scale": {t: 1.0 for t in spec.targets},
                            "rmse": {t: 1.0 for t in spec.targets}},
            "created_utc": utcnow(),
        }
        tmp = Path(out_root) / f".pca_ridge.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            meta["calibration"].update(self._calibrate(tmp, arrays, meta, snapshot))
            buf = io.BytesIO()
            np.savez(buf, **arrays)
            payload = buf.getvalue()
            meta["bundle_id"] = "pca_ridge-" + hashlib.sha256(
                payload + json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]
            (tmp / "arrays.npz").write_bytes(payload)
            (tmp / "bundle.json").write_text(json.dumps(meta, indent=1, default=str))
            final = Path(out_root) / meta["bundle_id"]
            if final.exists():
                shutil.rmtree(tmp)
            else:
                os.rename(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return final

    def _calibrate(self, tmp: Path, arrays: dict, meta: dict, snapshot: Snapshot) -> dict:
        """Error-bar scale and held-out RMSE per target, in the transformed metric."""
        split = "val" if len(snapshot.samples("val")) else "train"
        np.savez(tmp / "arrays.npz", **arrays)
        (tmp / "bundle.json").write_text(json.dumps(meta | {"bundle_id": "uncalibrated"}, default=str))
        bundle = PCARidgeBundle(tmp)
        data = snapshot.load(meta["inputs"] + meta["targets"], split)
        _, mean_flat, std_flat = bundle._raw({k: data[k] for k in meta["inputs"]})
        truth = bundle.out.forward(data)
        scale, rmse = {}, {}
        for name, sl in bundle.out.slices().items():
            err = np.abs(mean_flat[:, sl] - truth[:, sl]).mean(1)
            spread = std_flat[:, sl].mean(1)
            scale[name] = float(err.mean() / max(spread.mean(), 1e-12))
            rmse[name] = float(np.sqrt((err**2).mean())) or 1e-12
        return {"split": split, "errbar_scale": scale, "rmse": rmse}
