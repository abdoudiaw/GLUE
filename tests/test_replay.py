"""Replay event files for the B2.5 surrogate test, with the baseline learner."""
import json

import netCDF4
import numpy as np
from synth import write_case

from glue2 import baseline  # noqa: F401
from glue2.events import read_arrays
from glue2.ingest import ingest
from glue2.learner import load_bundle, make_learner
from glue2.replay import predict_event, write_replay_event
from glue2.snapshot import build_snapshot


def test_replay_files_replace_only_the_bundle_targets(catalog, spec, tmp_path):
    root = tmp_path / "cases"
    for i in range(30):
        write_case(root, i, average=False)
    ingest(catalog, [root], settle_seconds=0)
    snap = build_snapshot(catalog, spec, tmp_path / "snaps")
    bundle = load_bundle(make_learner("pca_ridge").fit(snap, tmp_path / "bundles"))
    template = next(e["path"] for e in snap.events("test"))
    pred = predict_event(bundle, template)
    assert pred.score.shape == (1,)

    out = {}
    for mode in ("surrogate", "masked_truth"):
        out[mode] = write_replay_event(bundle, template, tmp_path / f"{mode}.nc", mode, pred)
    truth = read_arrays(template, list(spec.targets) + ["braeir_dni"])
    with netCDF4.Dataset(tmp_path / "surrogate.nc") as ds:
        for name in spec.targets:
            np.testing.assert_allclose(ds.variables[name][...], pred.mean[name][0], rtol=1e-6)
        np.testing.assert_array_equal(ds.variables["braeir_dni"][...], truth["braeir_dni"])   # inputs untouched
        stamp = json.loads(ds.getncattr("glue2_replay"))
        assert stamp["bundle_id"] == bundle.bundle_id and stamp["mode"] == "surrogate"
    with netCDF4.Dataset(tmp_path / "masked_truth.nc") as ds:
        for name in spec.targets:
            got = ds.variables[name][...]
            keep = pred.mean[name][0] != 0
            np.testing.assert_array_equal(got[keep], truth[name][keep])
            assert not got[~keep].any()
    for name in spec.targets:
        assert out["masked_truth"]["variables"][name]["rel_l2"] <= out["surrogate"]["variables"][name]["rel_l2"] + 1e-9
    assert out["surrogate"]["variables"]["eirbra_sni"]["rel_l2"] < 0.2


def test_service_options_restrict_gate_and_keep_tallies(catalog, spec, tmp_path):
    """gate_score over chosen kinds; skipped targets keep the request's values; requests kept."""
    from glue2.surrogate_service import gate_score, serve
    root = tmp_path / "cases"
    for i in range(30):
        write_case(root, i, average=False)
    ingest(catalog, [root], settle_seconds=0)
    snap = build_snapshot(catalog, spec, tmp_path / "snaps")
    bundle = load_bundle(make_learner("pca_ridge").fit(snap, tmp_path / "bundles"))
    template = next(e["path"] for e in snap.events("test"))
    pred = predict_event(bundle, template)
    s_all, _ = gate_score(bundle, pred, None)
    s_sni, _ = gate_score(bundle, pred, ["sni"])
    assert s_all == float(pred.score[0]) and s_sni <= s_all
    run = tmp_path / "run"
    run.mkdir()
    import shutil
    shutil.copyfile(template, run / "eirene_surrogate_request.nc")
    serve(str(bundle.path), str(run), None, "always", str(tmp_path / "svc.jsonl"), once=True,
          gate_kinds=["sni", "see"], skip_targets=["wneutrals_*"], keep_requests=str(tmp_path / "keep"))
    truth = read_arrays(template, list(spec.targets))
    with netCDF4.Dataset(run / "eirene_surrogate_answer.nc") as ds:
        np.testing.assert_array_equal(ds.variables["wneutrals_wldna"][...], truth["wneutrals_wldna"])
        assert not np.array_equal(ds.variables["eirbra_sni"][...], truth["eirbra_sni"])
    rec = json.loads((tmp_path / "svc.jsonl").read_text().splitlines()[-1])
    assert set(rec["ratios"]) >= {"sni", "see"} and rec["surrogate"] is True
    assert (tmp_path / "keep" / "request_call000000.nc").exists()
