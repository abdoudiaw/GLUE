import json
import shutil

import netCDF4
from synth import mark_campaign_case, write_case

from glue2.ingest import ingest


def campaign_case(root, index, **kw):
    paths = write_case(root, index, average=False)
    mark_campaign_case(paths[0].parent, index, **kw)
    return paths[0].parent


def test_campaign_sidecars_quarantine_and_batches(catalog, tmp_path):
    root = tmp_path / "SOLPS_DB" / "eirene_training_v2__ens"
    dirs = [campaign_case(root, i) for i in range(5)]
    in_progress = campaign_case(root, 5, success=False)
    quarantine = root / "_quarantine_rebuilt" / dirs[0].name
    shutil.copytree(write_case(tmp_path / "other", 99, average=False)[0].parent, quarantine)
    corrupt = campaign_case(root, 6)
    with netCDF4.Dataset(next(corrupt.glob("*_0002.nc")), "a") as ds:   # changed after hashing
        ds.variables["eirbra_see"][0, 0, 0] = 1.0

    # Campaign cases need no settle time; the success marker says they are complete.
    first = ingest(catalog, [root], settle_seconds=1e9, max_new_cases=3)
    assert first.cases == 3 and first.more and len(first.added) == 6
    rest = ingest(catalog, [root], settle_seconds=1e9)
    assert rest.unsettled == 2                                    # no success marker yet
    assert [r for _, r in rest.rejected] == ["sha256 does not match the case manifest"]
    assert catalog.scalar("SELECT COUNT(*) FROM events") == 11
    assert catalog.scalar("SELECT COUNT(*) FROM events WHERE path LIKE '%_quarantine%'") == 0

    controls = json.loads(catalog.scalar("SELECT controls FROM cases WHERE case_id=?", (dirs[1].name,)))
    assert set(controls) == {"core.density_m-3", "power.Pe_W", "gas_puffing.targets.D2.value"}

    (in_progress / "EIRENE_TRAINING_SUCCESS").touch()
    assert len(ingest(catalog, [root], settle_seconds=1e9).added) == 2
