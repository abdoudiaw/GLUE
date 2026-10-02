"""The GLUE active-learning cycle for the B2.5-EIRENE seam.

One control-plane process owns the catalog writer and repeats:

1. ingest   - register new event files (campaign output, teacher answers);
2. retrain  - when enough new events exist, freeze a snapshot, fit a candidate
              bundle, and promote it only if it does not regress on the test
              split against the currently promoted bundle;
3. acquire  - score pool backgrounds with the promoted bundle and request
              teacher answers for those its gate rejects, most uncertain first.

Nothing here is on the B2 iteration path. A running SOLPS case loads one
promoted bundle at start-up and keeps it; promotion affects later runs only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glue2 import baseline  # noqa: F401  (registers the pca_ridge learner)
from glue2.catalog import Catalog, utcnow
from glue2.ingest import ingest
from glue2.learner import evaluate, load_bundle, make_learner
from glue2.snapshot import SnapshotSpec, build_snapshot
from glue2.teacher import Pool, Teacher


@dataclass
class LoopConfig:
    workdir: Path
    spec: SnapshotSpec
    ingest_roots: list[Path]
    learner: str = "pca_ridge"
    learner_params: dict = field(default_factory=dict)
    settle_seconds: float = 60.0
    retrain_min_new: int = 1           # new eligible events that trigger retraining
    batch_size: int = 4                # requests per cycle
    max_open_requests: int = 16
    promote_tolerance: float = 0.05    # allowed relative regression per held-out split

    @property
    def catalog_path(self) -> Path:
        return Path(self.workdir) / "catalog.sqlite"

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        d = dict(d)
        d["workdir"] = Path(d["workdir"])
        d["spec"] = SnapshotSpec.from_dict(d["spec"])
        d["ingest_roots"] = [Path(p) for p in d["ingest_roots"]]
        return cls(**d)


class GlueLoop:
    def __init__(self, config: LoopConfig, catalog: Catalog,
                 pool: Pool | None = None, teacher: Teacher | None = None):
        if not catalog.writable:
            raise PermissionError("the loop needs the catalog writer")
        self.cfg, self.catalog, self.pool, self.teacher = config, catalog, pool, teacher
        self.snapshot_root = Path(config.workdir) / "snapshots"
        self.bundle_root = Path(config.workdir) / "bundles"

    # ----------------------------------------------------------------- retrain

    def _trained_high_water(self) -> int:
        return int(self.catalog.scalar(
            "SELECT COALESCE(MAX(s.high_water), 0) FROM bundles b JOIN snapshots s USING (snapshot_id)"))

    def new_eligible(self) -> int:
        kinds = self.cfg.spec.event_kinds
        return int(self.catalog.scalar(
            f"SELECT COUNT(*) FROM events WHERE event_id > ? AND event_kind IN ({','.join('?' for _ in kinds)})",
            (self._trained_high_water(), *kinds)))

    def retrain(self) -> dict:
        snap = build_snapshot(self.catalog, self.cfg.spec, self.snapshot_root)
        bundle = load_bundle(make_learner(self.cfg.learner, **self.cfg.learner_params)
                             .fit(snap, self.bundle_root))
        metrics = {s: evaluate(bundle, snap, s) for s in ("val", "test", "acq_test")}

        current = self.catalog.promoted_bundle()
        decision, reason = "promoted", "no promoted bundle"
        if current is not None:
            incumbent = load_bundle(current["path"])
            metrics["incumbent"] = {s: evaluate(incumbent, snap, s) for s in ("test", "acq_test")}
            decision, reason = self._promotion(metrics)

        now = utcnow()
        with self.catalog.transaction() as conn:
            if decision == "promoted" and current is not None:
                conn.execute("UPDATE bundles SET status='retired' WHERE bundle_id=?", (current["bundle_id"],))
            conn.execute(
                "INSERT INTO bundles (bundle_id, snapshot_id, learner, path, status, metrics, created_utc, promoted_utc)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (bundle.bundle_id, snap.snapshot_id, self.cfg.learner, str(bundle.path), decision,
                 json.dumps(metrics), now, now if decision == "promoted" else None))
        self.catalog.audit("retrain", bundle_id=bundle.bundle_id, snapshot_id=snap.snapshot_id,
                           decision=decision, reason=reason)
        return {"bundle_id": bundle.bundle_id, "snapshot_id": snap.snapshot_id,
                "decision": decision, "reason": reason,
                "n_train": snap.manifest["counts"]["train"], "test": metrics["test"]}

    def _promotion(self, metrics: dict) -> tuple[str, str]:
        """Promote when the seed test split does not regress beyond the tolerance,
        the acquired hold-out does not regress either, and at least one improves."""
        tol = self.cfg.promote_tolerance
        if metrics["test"]["n"] == 0:
            return "rejected", "empty test split"
        verdicts, improved, parts = [], False, []
        for split in ("test", "acq_test"):
            cand, inc = metrics[split], metrics["incumbent"][split]
            if cand["n"] == 0:
                continue
            verdicts.append(cand["rel_l2"] <= inc["rel_l2"] * (1 + tol))
            improved |= cand["rel_l2"] < inc["rel_l2"]
            parts.append(f"{split} {cand['rel_l2']:.4g} vs {inc['rel_l2']:.4g}")
        reason = "; ".join(parts) + f" (tolerance {tol:g})"
        return ("promoted" if all(verdicts) and improved else "rejected"), reason

    # ----------------------------------------------------------------- acquire

    def acquire(self) -> dict:
        current = self.catalog.promoted_bundle()
        if self.pool is None or self.teacher is None or current is None:
            return {"requested": 0, "reason": "no pool, teacher, or promoted bundle"}
        known = {r[0] for r in self.catalog.query("SELECT DISTINCT background_hash FROM events")}
        known |= {r[0] for r in self.catalog.query("SELECT background_hash FROM requests")}
        cands = [c for c in self.pool.candidates() if c.candidate_id not in known]
        slots = self.cfg.max_open_requests - int(
            self.catalog.scalar("SELECT COUNT(*) FROM requests WHERE status='open'"))
        if not cands or slots <= 0:
            return {"requested": 0, "unlabelled": len(cands), "open_slots": max(slots, 0)}

        bundle = load_bundle(current["path"])
        pred = bundle.predict(self.pool.inputs(cands, bundle.meta["inputs"]))
        params = bundle.meta.get("params", {})
        urgency = np.maximum(pred.score / params.get("fussiness", 1.0),
                             pred.novelty / params.get("novelty_max", 1.0))
        order = [i for i in np.argsort(-urgency) if not pred.ok[i]][:min(self.cfg.batch_size, slots)]

        now = utcnow()
        requests = [{
            "request_id": f"req-{cands[i].candidate_id[:16]}",
            "background_hash": cands[i].candidate_id,
            "case_id": cands[i].case_id,
            "candidate_ref": cands[i].ref,
            "bundle_id": bundle.bundle_id,
            "score": json.dumps({"score": float(pred.score[i]), "novelty": float(pred.novelty[i]),
                                 "urgency": float(urgency[i]), "rank": rank}),
            "status": "open",
            "created_utc": now,
        } for rank, i in enumerate(order)]
        with self.catalog.transaction():
            for req in requests:
                self.catalog.insert("requests", req)
        if requests:
            self.teacher.submit(requests)
            self.catalog.audit("acquire", bundle_id=bundle.bundle_id,
                               request_ids=[r["request_id"] for r in requests])
        return {"requested": len(requests), "unlabelled": len(cands),
                "unlabelled_gate_rate": float(pred.ok.mean())}

    # ------------------------------------------------------------------- cycle

    def cycle(self) -> dict:
        rep = ingest(self.catalog, self.cfg.ingest_roots, settle_seconds=self.cfg.settle_seconds)
        out = {"utc": utcnow(), "ingest": rep.as_dict()}
        new = self.new_eligible()
        if new and new >= self.cfg.retrain_min_new:
            try:
                out["retrain"] = self.retrain()
            except ValueError as exc:          # e.g. too few training cases yet
                out["retrain"] = {"error": str(exc)}
        out["acquire"] = self.acquire()
        self.catalog.audit("cycle", **{k: v for k, v in out.items() if k != "utc"})
        return out
