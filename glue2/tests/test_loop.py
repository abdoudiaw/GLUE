import json
import subprocess
import sys
from pathlib import Path

import yaml
from synth import INPUTS, TARGETS, write_case

from glue2.catalog import Catalog
from glue2.learner import evaluate, load_bundle
from glue2.loop import GlueLoop, LoopConfig
from glue2.snapshot import Snapshot
from glue2.teacher import EventFilePool, SimulatedTeacher, SpoolTeacher


def make_config(tmp_path, spec, **kw):
    return LoopConfig(workdir=tmp_path / "work", spec=spec,
                      ingest_roots=[tmp_path / "seed", tmp_path / "inbox"],
                      settle_seconds=0, **kw)


def test_simulated_active_learning(tmp_path, spec):
    for i in range(16):                       # seed corpus: the existing campaign
        write_case(tmp_path / "seed", i, average=False)
    for i in range(100, 130):                 # unlabelled backgrounds, wider control range
        write_case(tmp_path / "pool", i, average=False, scale=1.0 + (i % 3) * 0.6)
    (tmp_path / "inbox").mkdir()
    cfg = make_config(tmp_path, spec, batch_size=4, max_open_requests=8)
    pool = EventFilePool(tmp_path / "pool")

    with Catalog.writer(cfg.catalog_path) as cat:
        loop = GlueLoop(cfg, cat, pool, SimulatedTeacher(pool, tmp_path / "inbox"))
        history = [loop.cycle() for _ in range(6)]
        summary = cat.summary()
        requests = cat.query("SELECT * FROM requests")
        bundles = cat.query("SELECT * FROM bundles ORDER BY created_utc")
        final_snapshot = Snapshot.open(cat.latest_snapshot()["path"])

    first = history[0]
    assert first["retrain"]["decision"] == "promoted"
    assert first["acquire"]["requested"] == 4                     # wide pool: the seed model is unsure
    assert sum(h["acquire"]["requested"] for h in history) == len(requests)
    assert len({r["background_hash"] for r in requests}) == len(requests)
    assert all(r["status"] == "fulfilled" for r in requests[:-4])  # last batch lands next cycle
    assert summary["events_by_origin"]["al_request"] >= 2 * (len(requests) - 4)
    statuses = [b["status"] for b in bundles]
    assert statuses.count("promoted") == 1 and statuses[0] == "retired"
    for b in bundles:                                             # every bundle is kept, immutable
        assert (Path(b["path"]) / "bundle.json").exists()

    # Acquisition pays off: on held-out acquired cases the promoted model beats the
    # seed-only model, without losing accuracy on the seed test split.
    seed_model = load_bundle(bundles[0]["path"])
    promoted = load_bundle(next(b["path"] for b in bundles if b["status"] == "promoted"))
    acq = [evaluate(m, final_snapshot, "acq_test") for m in (seed_model, promoted)]
    base = [evaluate(m, final_snapshot, "test") for m in (seed_model, promoted)]
    assert acq[0]["n"] > 0
    assert acq[1]["rel_l2"] < acq[0]["rel_l2"] / 5, acq
    assert base[1]["rel_l2"] <= base[0]["rel_l2"] * 1.05, base


def test_spool_teacher_and_cli(tmp_path, spec):
    for i in range(12):
        write_case(tmp_path / "seed", i, average=False)
    write_case(tmp_path / "pool", 500, average=False, scale=5.0)
    (tmp_path / "inbox").mkdir()
    raw = {
        "workdir": str(tmp_path / "work"), "settle_seconds": 0,
        "ingest_roots": [str(tmp_path / "seed"), str(tmp_path / "inbox")],
        "spec": {"inputs": list(INPUTS), "targets": list(TARGETS),
                 "val_fraction": spec.val_fraction, "test_fraction": spec.test_fraction},
        "pool": {"root": str(tmp_path / "pool")},
        "teacher": {"kind": "spool", "spool": str(tmp_path / "spool")},
    }
    config = tmp_path / "glue2.yaml"
    config.write_text(yaml.safe_dump(raw))

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "glue2", *args, "--config", str(config)],
                              capture_output=True, text=True, cwd=Path(__file__).parents[1])

    out = cli("cycle")
    assert out.returncode == 0, out.stderr
    cycle = json.loads(out.stdout.splitlines()[-1])
    assert cycle["acquire"]["requested"] == 1
    req = json.loads(next((tmp_path / "spool").glob("req-*.json")).read_text())
    assert req["case_id"] == "run_000001f4__D" and req["status"] == "open"

    cfg = LoopConfig.from_dict({k: v for k, v in raw.items() if k not in ("pool", "teacher")})
    with Catalog.writer(cfg.catalog_path):
        busy = cli("cycle")
        status = cli("status")                                     # readers never wait on the writer
    assert busy.returncode == 2 and "another writer" in busy.stderr
    assert status.returncode == 0 and json.loads(status.stdout)["requests"] == {"open": 1}
