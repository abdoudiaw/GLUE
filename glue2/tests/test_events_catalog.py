import os
import shutil
import sqlite3
from pathlib import Path

import netCDF4
import pytest
from synth import write_case

from glue2.catalog import Catalog, WriterBusy
from glue2.events import EventError, read_event_info
from glue2.ingest import ingest

REAL_EVENT = Path(os.environ.get(
    "GLUE2_REAL_EVENT", "~/Downloads/eirene_training_v2_b2call_00000000_single_call_0002.nc")).expanduser()


def test_event_info_and_background_hash(tmp_path):
    a = write_case(tmp_path, 1)
    b = write_case(tmp_path, 2)
    infos = [read_event_info(p) for p in a]
    assert {i.case_id for i in infos} == {"run_00000001__D"}
    assert [i.event_kind for i in infos] == ["single_call", "single_call", "average_used_by_b2"]
    assert len({i.background_hash for i in infos}) == 1          # repeats share the background
    assert len({i.sha256 for i in infos}) == 3                   # but are different answers
    assert read_event_info(b[0]).background_hash != infos[0].background_hash


def test_invalid_events_rejected(tmp_path):
    path = write_case(tmp_path, 1)[0]
    with netCDF4.Dataset(path, "a") as ds:
        ds.variables["eirbra_see"][0, 0, 0] = float("nan")
    with pytest.raises(EventError, match="non-finite"):
        read_event_info(path)
    bad = tmp_path / "eirene_training_v2_b2call_00000000_single_call_0009.nc"
    bad.write_bytes(b"partial")
    with pytest.raises(EventError, match="cannot open"):
        read_event_info(bad)
    with pytest.raises(EventError, match="filename"):
        read_event_info(shutil.copy(write_case(tmp_path, 3)[0], tmp_path / "x.nc"))


@pytest.mark.skipif(not REAL_EVENT.exists(), reason="real event file not available")
def test_real_event_file():
    info = read_event_info(REAL_EVENT)
    assert info.schema_version.startswith("2.")
    assert info.repeat_index == 2 and info.repeat_count == 2


def test_single_writer_and_concurrent_reader(tmp_path, campaign):
    path = tmp_path / "work" / "catalog.sqlite"
    with Catalog.writer(path) as cat:
        with pytest.raises(WriterBusy):
            Catalog.writer(path)
        ingest(cat, [campaign], settle_seconds=0)
        with Catalog.reader(path) as ro:
            assert ro.summary()["events"] == 36
            with pytest.raises(PermissionError):
                ro.audit("x")
            with pytest.raises(sqlite3.OperationalError):
                ro.conn.execute("DELETE FROM events")
    with Catalog.writer(path) as cat:                            # lock released on close
        assert cat.high_water() == 36


def test_ingest_idempotent_settle_reject_duplicate(catalog, campaign, tmp_path):
    newest = max(p.stat().st_mtime for p in campaign.rglob("*.nc"))
    rep = ingest(catalog, [campaign], settle_seconds=60, now=newest + 1)
    assert rep.unsettled == 36 and not rep.added                # still being written
    rep = ingest(catalog, [campaign], settle_seconds=0)
    assert len(rep.added) == 36
    assert not ingest(catalog, [campaign], settle_seconds=0).added

    bad = campaign / "run_bad__D" / "eirene_training_v2_b2call_00000000_single_call_0001.nc"
    bad.parent.mkdir()
    bad.write_bytes(b"truncated")
    assert len(ingest(catalog, [campaign], settle_seconds=0).rejected) == 1
    assert not ingest(catalog, [campaign], settle_seconds=0).rejected   # not retried while unchanged
    write_case(tmp_path / "fixed", 99)[0].replace(bad)                   # rewritten in place
    assert len(ingest(catalog, [campaign], settle_seconds=0).added) == 1
    assert catalog.scalar("SELECT COUNT(*) FROM rejected") == 0

    dup = tmp_path / "mirror" / "run_00000000__D"
    shutil.copytree(campaign / "run_00000000__D", dup)
    rep = ingest(catalog, [tmp_path / "mirror"], settle_seconds=0)
    assert rep.duplicates == 3 and not rep.added
    assert ingest(catalog, [tmp_path / "mirror"], settle_seconds=0).duplicates == 0


def test_request_fulfilment_links_late_repeats(catalog, tmp_path):
    paths = write_case(tmp_path / "answers", 7)
    info = read_event_info(paths[0])
    with catalog.transaction():
        catalog.insert("requests", {"request_id": "req-1", "background_hash": info.background_hash,
                                    "case_id": info.case_id, "candidate_ref": "x", "score": "{}",
                                    "status": "open", "created_utc": "now"})
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    shutil.copy2(paths[0], inbox / paths[0].name)
    rep = ingest(catalog, [inbox], settle_seconds=0)
    assert rep.fulfilled == ["req-1"]
    shutil.copy2(paths[1], inbox / paths[1].name)                # second repeat arrives later
    assert ingest(catalog, [inbox], settle_seconds=0).fulfilled == []
    rows = catalog.query("SELECT origin, request_id FROM events")
    assert [tuple(r) for r in rows] == [("al_request", "req-1")] * 2
    assert catalog.scalar("SELECT status FROM requests") == "fulfilled"


def test_writer_adds_used_by_b2_to_an_older_catalog(tmp_path):
    path = tmp_path / "catalog.sqlite"
    Catalog.writer(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE events DROP COLUMN used_by_b2")
    with Catalog.writer(path) as cat:
        assert "used_by_b2" in {row[1] for row in cat.query("PRAGMA table_info(events)")}
