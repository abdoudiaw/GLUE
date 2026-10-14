"""Job specs render to complete scripts without touching any host."""
import re
from pathlib import Path

import pytest

from glue2 import jobs

SPECS = sorted((jobs.JOBS / "specs").glob("*.yaml"))


@pytest.mark.parametrize("spec", SPECS, ids=[s.stem for s in SPECS])
def test_every_spec_renders(spec, tmp_path):
    out = jobs.render(spec, tmp_path)
    text = out.read_text()
    assert text.startswith("#!")
    left = sorted(set(re.findall(r"\$\{([a-z][_a-z0-9]*)\}", text)) - {"launch_env_tcsh"})
    # shell's own ${...} expansions in the templates are upper case or have modifiers; driver
    # placeholders are lower case and must all be substituted
    assert not left, f"unsubstituted driver variables in {out.name}: {left}"
    assert "${launch_env_tcsh}" not in text


def test_couple_spec_has_its_case_and_service():
    out = jobs.render(jobs.JOBS / "specs" / "mora_couple_gate10_transient.yaml", Path("/tmp/glue2_jobs_test"))
    text = out.read_text()
    assert "b2fstate_step00050" in text and "al_tests/gate10_transient" in text
    assert "glue2.surrogate_service" in text and "'eirene_training_replay'      '3'" in text
    assert "-np 32" in text                                   # host default leaves headroom


def test_couple_needs_bundle_for_surrogate_mode(tmp_path):
    spec = tmp_path / "bad.yaml"
    spec.write_text("name: x\nstage: couple\nhost: mora\nparams: {init_state: /a, replay: 3}\n")
    with pytest.raises(ValueError, match="bundle"):
        jobs.render(spec, tmp_path)
