"""SOLSTICE sources learner: a seed ensemble of the SOLSTICE `sources` GNN behind
the GLUE2 Learner/Bundle protocol (mean, errbar, score, novelty, ok).

* fit() trains `members` seeds with `solstice.training.cli` (task `sources`, the
  same command `glue2 train` runs) on the snapshot's train split, or wraps
  already trained run directories (`runs=[...]`), and writes a bundle directory
  holding one `member_<k>.pt` per seed, `arrays.npz` and `bundle.json`.
* predict() rebuilds the trainer's input pipeline (interior cells, cell order of
  the mesh store, per-channel transforms, standardized log stratum strengths),
  decodes every member to native units and returns their mean in the event
  file's layout (guard cells and inactive strata zero), so `glue2.learner.evaluate`
  compares it with the snapshot targets directly. Members trained with the
  wall-tally head also return the six per-stratum wall tallies (slot 0 of the
  event array is the sum over strata); bundles without it predict the sources only.
* errbar per source kind is the member spread in the trainer's normalized target
  metric, scaled on the validation split so it matches the actual error on
  average; `rmse` is the held-out error in the same metric; `score` = max over
  kinds of errbar / rmse and the gate accepts `score <= fussiness` (GLUE's iserrok).
* novelty is the mean k-NN distance of a case summary (per-case mean and spread
  of each normalized plasma channel, plus the standardized log stratum strengths
  that vary in training) to the training cases, divided by the largest such
  leave-one-out distance among training cases; `novelty > novelty_max` rejects.

torch and solstice are imported lazily, so registering the learner costs nothing
for a GLUE2 process that only runs the baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from glue2.catalog import utcnow
from glue2.learner import Prediction, register
from glue2.snapshot import Snapshot

NAME = "solstice_sources"
SOURCES = ("eirbra_sni", "eirbra_smo", "eirbra_see", "eirbra_sei")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _kinds(target_names: list[str]) -> list[str]:
    out = []
    for n in target_names:
        k = n.split(":")[0]
        if k not in out:
            out.append(k)
    return out


class _Member:
    """One trained seed: model, normalizers and latent graph from its best.pt."""

    def __init__(self, pt_path: Path, mesh, device: str):
        import torch
        from solstice.models import build_model
        from solstice.training import sources_data as sd
        from solstice.training.data import MeshGraph

        ck = torch.load(pt_path, map_location="cpu", weights_only=False)
        if ck.get("task") != "sources":
            raise ValueError(f"{pt_path} is not a sources checkpoint")
        self.ck = ck
        self.in_norm = sd.NodeNorm.from_dict(ck)
        self.out_norm = sd.CellTargetNorm.from_dict(ck)
        self.raw_input_names = list(ck["raw_input_names"])
        self.target_names = list(ck["target_names"])
        self.strata = list(ck["strata"])
        self.perm = np.asarray(ck["cell_perm"])
        self.params = ck.get("params", "strata_weights")
        self.p_mean = None if ck.get("p_mean") is None else np.asarray(ck["p_mean"])
        self.p_std = None if ck.get("p_std") is None else np.asarray(ck["p_std"])
        self.tally_names = list(ck.get("tally_names") or [])
        self.g_norm = sd.CellTargetNorm.from_dict(ck["tally_norm"]) if ck.get("tally_norm") else None
        self.n_surfaces = int(ck["n_surfaces"]) if ck.get("n_surfaces") else 0
        self.graph = MeshGraph(mesh, n_latent=int(ck["n_latent"]), k_nn=int(ck["k_nn"]), seed=int(ck.get("seed", 0)),
                               node_features=str(ck.get("node_features", "extended")),
                               k_enc=int(ck.get("k_enc", 1)), k_dec=ck.get("k_dec"),
                               multimesh=list(ck.get("multimesh") or []))
        self.model = build_model(ck.get("model_class", "gnn"), ck["config"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()
        self.device = device

    def predict(self, X: np.ndarray, W: np.ndarray, batch: int = 16) -> tuple[np.ndarray, np.ndarray | None]:
        """Raw inputs (case, cell, raw channel) and strengths (case, stratum) ->
        native-unit sources (case, cell, channel) and wall tallies (case, surface,
        tally channel) or None for a member without the tally head."""
        from solstice.training.evaluate import predict
        from solstice.training.sources_cli import strata_params

        Nn = self.in_norm.transform(X, self.raw_input_names)
        if self.params == "strata_weights":
            Xp = strata_params(W, self.p_mean, self.p_std)[0]
        else:
            Xp = np.zeros((len(X), 1), np.float32)
        Yn, Gn = predict(self.model, self.graph, Xp, self.device, batch, Nn=Nn, with_global=True)
        G = None
        if self.g_norm is not None and Gn is not None:
            G = self.g_norm.inverse(Gn.reshape(len(X), self.n_surfaces, -1))
        return self.out_norm.inverse(Yn), G


class SolsticeSourcesBundle:
    def __init__(self, path: Path, device: str | None = None):
        self.path = Path(path)
        self.meta = json.loads((self.path / "bundle.json").read_text())
        self.bundle_id = self.meta["bundle_id"]
        self.device = device or self.meta.get("device") or _default_device()
        self._members: list[_Member] | None = None
        a = np.load(self.path / "arrays.npz")
        self.train_feats, self.feat_scale = a["train_feats"], a["feat_scale"]
        cal = self.meta["calibration"]
        self.errbar_scale = {k: float(v) for k, v in cal["errbar_scale"].items()}
        self.rmse = {k: float(v) for k, v in cal["rmse"].items()}
        self.knn_ref = float(cal["knn_ref"])
        p = self.meta["params"]
        self.fussiness, self.novelty_max, self.k = float(p["fussiness"]), float(p["novelty_max"]), int(p["knn"])
        lay = self.meta["layout"]
        self.bra_inputs, self.ny, self.nx = list(lay["bra_inputs"]), int(lay["ny_bra"]), int(lay["nx_bra"])
        self.keep = [int(k) for k in lay["keep"]]
        self.target_names, self.strata = list(self.meta["target_names"]), list(self.meta["strata"])
        self.tally_names = list(self.meta.get("tally_names") or [])
        self.tally_targets = list(self.meta.get("tally_targets") or [])
        self.kinds = _kinds(self.target_names) + _kinds(self.tally_names)

    @property
    def mesh_store(self) -> str:
        """The bundle records the mesh store it was built with; on another machine
        GLUE2_MESH_STORE, or a file of the same name under SOLSTICE_DATA, replaces it."""
        rec = Path(self.meta["mesh_store"])
        for cand in (os.environ.get("GLUE2_MESH_STORE"),
                     str(Path(os.environ["SOLSTICE_DATA"]) / rec.name) if os.environ.get("SOLSTICE_DATA") else None,
                     str(rec)):
            if cand and Path(cand).exists():
                return cand
        raise FileNotFoundError(f"mesh store {rec} not found; set GLUE2_MESH_STORE or SOLSTICE_DATA")

    # ---- members -------------------------------------------------------------------------
    @property
    def members(self) -> list[_Member]:
        if self._members is None:
            from solstice.training import sources_data as sd
            mesh = sd.load_mesh_store(self.mesh_store)
            self._members = [_Member(self.path / m, mesh, self.device) for m in self.meta["members"]]
        return self._members

    # ---- raw event layout <-> trainer arrays ----------------------------------------------
    def arrays(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Snapshot-layout inputs -> X (case, cell, raw channel), W (case, stratum)."""
        from solstice.training.sources_data import to_cells
        cols = []
        for name in self.bra_inputs:
            arr = np.asarray(inputs[name])
            while arr.ndim > 3:
                arr = arr[:, 0]
            cols.append(arr[:, 1:-1, 1:self.nx - 1])
        planes = np.stack(cols, axis=1).astype(np.float32)
        perm = self.members[0].perm
        X = to_cells(planes, perm)
        W = (np.asarray(inputs["b2_tflux"]) * np.asarray(inputs["b2_flux_scale"]))[:, self.keep].astype(np.float32)
        return X, W

    def embed(self, Y: np.ndarray, G: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """(case, cell, channel) native sources (and (case, surface, tally channel)
        tallies) -> {eirbra_* / wneutrals_wld*: (case, *shape)} as in the events."""
        n = len(Y)
        perm = self.members[0].perm
        out = {}
        for tname in self.tally_targets if G is not None else []:
            tally = tname.split("_", 1)[1]
            full = np.zeros((n, *self.meta["shapes"][tname]), np.float32)
            for j, k in enumerate(self.keep):
                ch = self.tally_names.index(f"{tally}:{self.strata[j]}")
                (full[:, k + 1, 0] if full.ndim == 4 else full[:, k + 1])[:] = G[:, :, ch]
            (full[:, 0, 0] if full.ndim == 4 else full[:, 0])[:] = full[:, 1:].sum(1)[:, 0] if full.ndim == 4 else full[:, 1:].sum(1)
            out[tname] = full
        for tname in self.meta["targets"]:
            if tname in self.tally_targets:
                continue
            kind = tname.split("_", 1)[1]
            shape = tuple(self.meta["shapes"][tname])
            full = np.zeros((n, *shape), np.float32)
            for j, k in enumerate(self.keep):
                ch = self.target_names.index(f"{kind}:{self.strata[j]}")
                flat = np.zeros((n, (self.ny - 2) * (self.nx - 2)), np.float32)
                flat[:, perm] = Y[:, :, ch]
                plane = flat.reshape(n, self.ny - 2, self.nx - 2)
                if full.ndim == 5:
                    full[:, k, 0, 1:-1, 1:self.nx - 1] = plane
                else:
                    full[:, k, 1:-1, 1:self.nx - 1] = plane
            out[tname] = full
        return out

    # ---- ensemble ------------------------------------------------------------------------
    def _raw(self, inputs: dict[str, np.ndarray]):
        X, W = self.arrays(inputs)
        outs = [m.predict(X, W) for m in self.members]
        preds = np.stack([y for y, _ in outs])                              # (member, case, cell, ch)
        mean, std = preds.mean(0), preds.std(0)
        gmean = gstd = None
        if self.tally_targets and all(g is not None for _, g in outs):
            gp = np.stack([g for _, g in outs])                             # (member, case, surface, ch)
            gmean, gstd = gp.mean(0), gp.std(0)
        feats = self.features(X, W)
        return X, W, (mean, gmean), (std, gstd), feats

    def normalized(self, arr: np.ndarray, tallies: bool = False) -> np.ndarray:
        """Native (case, cell, channel) -> the trainer's standardized metric (masked);
        tallies=True for (case, surface, tally channel)."""
        n0 = self.members[0].g_norm if tallies else self.members[0].out_norm
        return (arr / n0.std) * n0.mask

    def kind_stat(self, arr_n: np.ndarray, garr_n: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Mean over the masked entries of each source kind (and tally) -> {kind: (case,)}."""
        out = {}
        for names, norm, arr in ((self.target_names, self.members[0].out_norm, arr_n),
                                 (self.tally_names, self.members[0].g_norm, garr_n)):
            if arr is None or norm is None:
                continue
            for kind in _kinds(names):
                ch = [j for j, t in enumerate(names) if t.startswith(kind + ":")]
                m = norm.mask[:, ch]
                out[kind] = (arr[:, :, ch] * m).sum((1, 2)) / max(m.sum(), 1.0)
        return out

    def features(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        from solstice.training.sources_cli import strata_params
        m0 = self.members[0]
        Nn = m0.in_norm.transform(X, m0.raw_input_names)
        parts = [Nn.mean(1), Nn.std(1)]
        if m0.p_std is not None:
            var = m0.p_std > 1e-3
            if var.any():
                parts.append(strata_params(W, m0.p_mean, m0.p_std)[0][:, var])
        return np.concatenate(parts, axis=1).astype(np.float64)

    def novelty(self, feats: np.ndarray) -> np.ndarray:
        Z = feats / self.feat_scale
        d = np.sqrt(((Z[:, None, :] - self.train_feats[None]) ** 2).sum(-1))
        k = min(self.k, self.train_feats.shape[0])
        return np.sort(d, axis=1)[:, :k].mean(1) / self.knn_ref

    def predict(self, inputs: dict[str, np.ndarray]) -> Prediction:
        _, _, (mean, gmean), (std, gstd), feats = self._raw(inputs)
        spread = self.kind_stat(self.normalized(std), None if gstd is None else self.normalized(gstd, True))
        n = len(mean)
        kinds = [k for k in self.kinds if k in spread and k in self.errbar_scale]
        errbar, ratio = {}, np.zeros((n, len(kinds)))
        for j, kind in enumerate(kinds):
            eb = self.errbar_scale[kind] * spread[kind]
            errbar[("wneutrals_" if kind.startswith("wld") else "eirbra_") + kind] = eb
            ratio[:, j] = eb / max(self.rmse[kind], 1e-12)
        novelty = self.novelty(feats)
        score = ratio.max(1)
        ok = (score <= self.fussiness) & (novelty <= self.novelty_max)
        return Prediction(self.embed(mean, gmean), errbar, score, novelty, ok)


def _default_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_bundle(run_dirs: list[str | Path], snapshot: Snapshot, out_root: str | Path, mesh_store: str | Path,
                 params: dict, device: str | None = None) -> Path:
    """Wrap trained seed runs (each with best.pt) into a calibrated bundle directory."""
    device = device or _default_device()
    runs = [Path(r) for r in run_dirs]
    for r in runs:
        if not (r / "best.pt").exists():
            raise FileNotFoundError(f"{r}/best.pt missing")
    spec, shapes = snapshot.spec, snapshot.shapes
    bra = [n for n in spec.inputs if n.startswith("braeir_") and len(shapes[n]) >= 2]
    ny_bra, nx_bra = shapes[bra[0]][-2], shapes[bra[0]][-1]
    import torch
    ck0 = torch.load(runs[0] / "best.pt", map_location="cpu", weights_only=False)
    from glue2.b2view import _stratum_labels
    labels = _stratum_labels(snapshot, shapes["b2_tflux"][0])
    keep = [labels.index(s) for s in ck0["strata"]]
    targets = [t for t in SOURCES if t in spec.targets]
    tally_names = list(ck0.get("tally_names") or [])
    tally_targets = [t for t in spec.targets if t.startswith("wneutrals_wld")] if tally_names else []
    meta = {
        "learner": NAME, "snapshot_id": snapshot.snapshot_id,
        "inputs": list(spec.inputs), "targets": targets + tally_targets,
        "tally_targets": tally_targets, "tally_names": tally_names,
        "shapes": {k: list(v) for k, v in shapes.items() if k in targets + tally_targets},
        "layout": {"bra_inputs": bra, "ny_bra": int(ny_bra), "nx_bra": int(nx_bra), "keep": keep},
        "target_names": list(ck0["target_names"]), "strata": list(ck0["strata"]),
        "mesh_store": str(Path(mesh_store).resolve()), "device": device,
        "members": [f"member_{k}.pt" for k in range(len(runs))],
        "member_runs": [str(r.resolve()) for r in runs],
        "member_seeds": [], "member_epochs": [], "code_git": ck0.get("code_git"),
        "params": params, "n_train": len(snapshot.samples("train")),
        "calibration": {"knn_ref": 1.0, "errbar_scale": {}, "rmse": {}},
        "created_utc": utcnow(),
    }
    tmp = Path(out_root) / f".{NAME}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        h = hashlib.sha256()
        for k, r in enumerate(runs):
            shutil.copy2(r / "best.pt", tmp / f"member_{k}.pt")
            h.update((tmp / f"member_{k}.pt").read_bytes())
            ck = torch.load(r / "best.pt", map_location="cpu", weights_only=False)
            meta["member_seeds"].append(int(ck.get("seed", k)))
            meta["member_epochs"].append(int(ck.get("epoch", -1)))
        np.savez(tmp / "arrays.npz", train_feats=np.zeros((1, 1)), feat_scale=np.ones(1))
        (tmp / "bundle.json").write_text(json.dumps(meta | {"bundle_id": "uncalibrated"}, default=str))
        b = SolsticeSourcesBundle(tmp, device)

        # novelty reference from the training cases
        tr = snapshot.load(meta["inputs"], "train")
        feats = b.features(*b.arrays(tr))
        feat_scale = feats.std(0) + 1e-12
        Z = feats / feat_scale
        d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        k = min(int(params["knn"]), len(Z) - 1)
        knn_ref = float(np.sort(d, axis=1)[:, :k].mean(1).max())
        np.savez(tmp / "arrays.npz", train_feats=Z, feat_scale=feat_scale)
        meta["calibration"]["knn_ref"] = knn_ref
        b.train_feats, b.feat_scale, b.knn_ref = Z, feat_scale, knn_ref

        # error-bar scale and held-out error per source kind, in the normalized metric;
        # calibrated on the concatenation of `params["calib_splits"]` (default: val), so a
        # transient regime held out in acq_test can enter the calibration when asked for
        from glue2.b2view import training_arrays
        from solstice.training.sources_data import stack_tallies, to_cells
        want = list(params.get("calib_splits") or ["val"])
        splits = [sp for sp in want if len(snapshot.samples(sp))] or ["train"]
        err_parts, spread_parts = [], []
        for split in splits:
            data = snapshot.load(meta["inputs"] + targets, split)
            X, W, (mean, gmean), (std, gstd), _ = b._raw({k: data[k] for k in meta["inputs"]})
            t = training_arrays(snapshot, split, list(ck0["strata"]))
            Y_true = to_cells(t.targets, b.members[0].perm)
            gerr = gspread = None
            if gmean is not None:
                G_true, _ = stack_tallies(t.tallies, t.strata)
                gerr, gspread = b.normalized(np.abs(gmean - G_true), True), b.normalized(gstd, True)
            err_parts.append(b.kind_stat(b.normalized(np.abs(mean - Y_true)), gerr))
            spread_parts.append(b.kind_stat(b.normalized(std), gspread))
        err = {kd: np.concatenate([e[kd] for e in err_parts]) for kd in err_parts[0]}
        spread = {kd: np.concatenate([e[kd] for e in spread_parts]) for kd in spread_parts[0]}
        meta["calibration"].update({
            "split": "+".join(splits),
            "errbar_scale": {kd: float(err[kd].mean() / max(spread[kd].mean(), 1e-12)) for kd in err},
            "rmse": {kd: float(np.sqrt((err[kd] ** 2).mean())) or 1e-12 for kd in err},
        })
        h.update(json.dumps(meta, sort_keys=True, default=str).encode())
        meta["bundle_id"] = f"{NAME}-" + h.hexdigest()[:12]
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


@register(NAME, SolsticeSourcesBundle)
class SolsticeSourcesLearner:
    name = NAME

    def __init__(self, members: int = 5, epochs: int = 1500, max_hours: float = 3.8,
                 train_config: str = "external/solstice/configs/training/eirene_sources.yaml",
                 mesh_store: str | None = None, library_src: str = "external/solstice/src",
                 fussiness: float = 1.5, novelty_max: float = 1.0, knn: int = 5,
                 device: str | None = None, runs: list[str] | None = None, overrides: list[str] | None = None,
                 warm_start: str | None = None, calib_splits: list[str] | None = None, lr: float | None = None,
                 train_jobs: int = 1, train_devices: list[str] | None = None, train_prefix: str = ""):
        """warm_start: a bundle directory whose members are fine-tuned (one run per member,
        `train.init_from` = that member, `init_strict` false so a new head starts fresh) instead
        of training `members` seeds from scratch; calib_splits: splits the error bar is calibrated
        on (default val); lr: learning rate override (fine-tunes use a lower one); train_jobs: members
        trained concurrently, each on the next of train_devices (e.g. ["cuda:1", "cuda:2", "cuda:3"]);
        train_prefix: launcher words put before each trainer command (e.g. an srun step onto the node
        that runs the coupled case, so the retrain uses its idle GPUs)."""
        self.params = dict(members=members, epochs=epochs, max_hours=max_hours, train_config=train_config,
                           mesh_store=mesh_store, library_src=library_src, fussiness=fussiness,
                           novelty_max=novelty_max, knn=knn, overrides=list(overrides or []),
                           warm_start=warm_start, calib_splits=list(calib_splits) if calib_splits else None, lr=lr,
                           train_jobs=int(train_jobs), train_devices=list(train_devices or []), train_prefix=train_prefix)
        self.device, self.runs = device, runs

    def training_commands(self, snapshot: Snapshot, out_root: str | Path) -> list[tuple[Path, list[str]]]:
        """(run directory, trainer command) per member, without running anything."""
        p = self.params
        mesh_store = REPO_ROOT / p["mesh_store"]
        warm = Path(p["warm_start"]) if p.get("warm_start") else None
        if warm:
            meta = json.loads((warm / "bundle.json").read_text())
            members = [(k, int(seed), warm / m) for k, (m, seed) in enumerate(zip(meta["members"], meta["member_seeds"]))]
        else:
            members = [(k, k, None) for k in range(int(p["members"]))]
        out = []
        for k, seed, init in members:
            out_dir = Path(out_root) / ".runs" / snapshot.snapshot_id / f"member{k}"
            cmd = [sys.executable, "-m", "solstice.training.cli", "--config", str(REPO_ROOT / p["train_config"]),
                   "--set", f"data.snapshot={snapshot.path}", "--set", f"data.mesh_store={mesh_store}",
                   "--set", f"out_dir={out_dir}", "--set", f"seed={seed}",
                   "--set", f"train.epochs={p['epochs']}", "--set", f"train.max_hours={p['max_hours']}"]
            if init is not None:
                cmd += ["--set", f"train.init_from={init}", "--set", "train.init_strict=false"]
            if p.get("lr"):
                cmd += ["--set", f"train.lr={p['lr']}"]
            for item in p["overrides"]:
                cmd += ["--set", item]
            if self.device:
                cmd += ["--device", self.device]
            out.append((out_dir, cmd))
        return out

    def fit(self, snapshot: Snapshot, out_root: str | Path) -> Path:
        p = self.params
        if not p["mesh_store"]:
            raise ValueError("solstice_sources needs mesh_store (learner_params or train.mesh_store)")
        mesh_store = REPO_ROOT / p["mesh_store"]
        runs = [Path(r) for r in self.runs] if self.runs else []
        if not runs:
            lib_src = REPO_ROOT / p["library_src"]
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(lib_src), str(REPO_ROOT), env.get("PYTHONPATH", "")])
            jobs = max(1, int(p.get("train_jobs") or 1))
            devices = list(p.get("train_devices") or [])
            prefix = shlex.split(p.get("train_prefix") or "")
            running: list[tuple[Path, subprocess.Popen]] = []

            def reap(block: bool) -> None:
                while running and (block or len(running) >= jobs):
                    out_dir, proc = running[0]
                    rc = proc.wait()
                    running.pop(0)
                    if rc != 0:
                        for _, other in running:
                            other.kill()
                        raise RuntimeError(f"training of {out_dir.name} failed (exit {rc})")

            for k, (out_dir, cmd) in enumerate(self.training_commands(snapshot, out_root)):
                runs.append(out_dir)
                if (out_dir / "metrics.json").exists():
                    continue
                if devices:
                    dev = devices[k % len(devices)]
                    cmd = [c for c in cmd]
                    if "--device" in cmd:
                        cmd[cmd.index("--device") + 1] = dev
                    else:
                        cmd += ["--device", dev]
                reap(block=False)
                print("solstice_sources fit:", " ".join(prefix + cmd), flush=True)
                running.append((out_dir, subprocess.Popen(prefix + cmd, env=env, cwd=REPO_ROOT)))
            reap(block=True)
        return build_bundle(runs, snapshot, out_root, mesh_store, p, self.device)


def main(argv=None) -> int:
    """Build a bundle from trained runs and score it on the snapshot splits.

        python -m glue2.solstice_learner --snapshot <dir> --mesh-store <nc> --out-root <dir> runs/seed0 runs/seed1 ...
    """
    import argparse
    from glue2.learner import evaluate, load_bundle
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--mesh-store", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fussiness", type=float, default=1.5)
    ap.add_argument("--novelty-max", type=float, default=1.0)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--calib-splits", default="val",
                    help="comma-separated snapshot splits the error bar is calibrated on (default: val)")
    a = ap.parse_args(argv)
    snap = Snapshot.open(a.snapshot)
    params = dict(members=len(a.runs), fussiness=a.fussiness, novelty_max=a.novelty_max, knn=a.knn,
                  mesh_store=a.mesh_store, source="existing runs",
                  calib_splits=[sp.strip() for sp in a.calib_splits.split(",") if sp.strip()])
    path = build_bundle(a.runs, snap, a.out_root, a.mesh_store, params, a.device)
    b = load_bundle(path)
    print(f"bundle {b.bundle_id} -> {path}")
    print("calibration:", json.dumps(b.meta["calibration"]))
    for split in ("val", "test"):
        m = evaluate(b, snap, split)
        print(f"== {split} ({m['n']} cases): mean rel_l2 {m['rel_l2']:.3f}, gate accepts {100 * m['gate_rate']:.0f} %")
        for t, v in m["per_target"].items():
            print(f"   {t:10s} rel_l2 {v['rel_l2']:.3f}  sum_rel {v['sum_rel']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
