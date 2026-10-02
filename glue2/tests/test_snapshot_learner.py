import numpy as np
import pytest
from synth import background, case_controls, write_case

from glue2 import baseline  # noqa: F401
from glue2.events import read_arrays
from glue2.ingest import ingest
from glue2.learner import evaluate, load_bundle, make_learner
from glue2.snapshot import Snapshot, build_snapshot


def test_snapshot_is_frozen_grouped_and_faithful(catalog, campaign, spec, tmp_path):
    ingest(catalog, [campaign], settle_seconds=0)
    root = tmp_path / "snaps"
    snap = build_snapshot(catalog, spec, root)
    m = snap.manifest
    assert sum(m["counts"].values()) == 24                       # single_call only
    split_of = {}
    for e in m["events"]:
        assert split_of.setdefault(e["case_id"], e["split"]) == e["split"]
    assert all(m["counts"][s] for s in ("train", "val", "test"))

    first = m["events"][0]
    src = read_arrays(first["path"], ["eirbra_see"])["eirbra_see"]
    got = snap.load(["eirbra_see"])["eirbra_see"][0]
    np.testing.assert_allclose(got, src.astype(np.float32))

    assert build_snapshot(catalog, spec, root).snapshot_id == snap.snapshot_id
    write_case(campaign, 50)
    ingest(catalog, [campaign], settle_seconds=0)
    newer = build_snapshot(catalog, spec, root)
    assert newer.snapshot_id != snap.snapshot_id
    assert Snapshot.open(snap.path).manifest == m                # old snapshot untouched
    old_splits = {e["case_id"]: e["split"] for e in m["events"]}
    assert all(old_splits.get(e["case_id"], e["split"]) == e["split"] for e in newer.manifest["events"])
    hw = build_snapshot(catalog, spec, root, high_water=snap.manifest["high_water"])
    assert hw.snapshot_id == snap.snapshot_id


def test_active_learning_cases_are_training_only(catalog, campaign, spec, tmp_path):
    from glue2.events import read_event_info
    extra = write_case(tmp_path / "answers", 77)
    bg = read_event_info(extra[0]).background_hash
    with catalog.transaction():
        catalog.insert("requests", {"request_id": "r", "background_hash": bg, "case_id": "run_0000004d__D",
                                    "candidate_ref": "x", "score": "{}", "status": "open",
                                    "created_utc": "now"})
    ingest(catalog, [campaign, tmp_path / "answers"], settle_seconds=0)
    snap = build_snapshot(catalog, spec, tmp_path / "snaps")
    assert {e["split"] for e in snap.events() if e["origin"] == "al_request"} == {"train"}


@pytest.fixture
def trained(catalog, spec, tmp_path):
    root = tmp_path / "big"
    for i in range(40):
        write_case(root, i, average=False)
    ingest(catalog, [root], settle_seconds=0)
    snap = build_snapshot(catalog, spec, tmp_path / "snaps")
    path = make_learner("pca_ridge").fit(snap, tmp_path / "bundles")
    return snap, load_bundle(path)


def test_baseline_learns_and_gates(trained):
    snap, bundle = trained
    metrics = evaluate(bundle, snap, "test")
    assert metrics["n"] > 0
    assert metrics["rel_l2"] < 0.1, metrics
    data = snap.load(list(snap.spec.targets), "test")
    mean_pred = {k: np.broadcast_to(snap.load([k], "train")[k].mean(0), data[k].shape) for k in data}
    mean_err = np.median([np.linalg.norm(mean_pred[k] - data[k]) / np.linalg.norm(data[k]) for k in data])
    assert metrics["rel_l2"] < 0.5 * mean_err                     # real skill over the mean predictor
    assert metrics["gate_rate"] > 0.5

    far = {k: v[None] for k, v in background(case_controls(3, scale=6.0)).items()}
    pred = bundle.predict({k: far[k] for k in bundle.meta["inputs"]})
    assert pred.novelty[0] > 1 and not pred.ok[0]


def test_bundle_reload_is_identical(trained):
    snap, bundle = trained
    inputs = snap.load(bundle.meta["inputs"], "test")
    again = load_bundle(bundle.path)
    a, b = bundle.predict(inputs), again.predict(inputs)
    for k in a.mean:
        np.testing.assert_array_equal(a.mean[k], b.mean[k])
    np.testing.assert_array_equal(a.ok, b.ok)
