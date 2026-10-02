import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synth import INPUTS, TARGETS, write_case  # noqa: E402

from glue2.catalog import Catalog  # noqa: E402
from glue2.snapshot import SnapshotSpec  # noqa: E402


@pytest.fixture
def spec():
    return SnapshotSpec(inputs=INPUTS, targets=TARGETS, val_fraction=0.2, test_fraction=0.2)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    for i in range(12):
        write_case(root, i)
    return root


@pytest.fixture
def catalog(tmp_path):
    with Catalog.writer(tmp_path / "work" / "catalog.sqlite") as cat:
        yield cat
