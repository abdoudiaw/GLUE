"""Job driver: one entry point for the build, campaign, train and couple stages.

A job spec (YAML) names a stage, a host and parameters; the host file
(`jobs/hosts/<host>.yaml`) holds the paths and launch environment of that
machine; the template (`jobs/templates/<stage>.<host>.<sh|csh>`) is rendered
with both (``${name}`` placeholders; shell variables without braces are left
alone). The rendered script is what runs, so it can always be read, copied
and launched by hand.

    python -m glue2.jobs render jobs/specs/mora_couple_gate50.yaml
    python -m glue2.jobs run    jobs/specs/mora_couple_gate50.yaml      # copy to the host, launch, print the log path
    python -m glue2.jobs status jobs/specs/mora_couple_gate50.yaml      # tail the job log on the host
    python -m glue2.jobs sync   mora                                     # rsync this package and SOLSTICE to the host

Spec:

    name: gate50_transient
    stage: couple            # build | campaign | couple | train
    host: mora               # jobs/hosts/mora.yaml
    params: {...}            # stage parameters, see STAGE_DEFAULTS

`run` launches with nohup/setsid under the host's jobs root and returns at
once; a SOLPS run on Mora starves sshd, so poll `status` sparingly. Perlmutter
jobs are rendered only: the user submits them (MFA login).
"""

from __future__ import annotations

import argparse
import shlex
import string
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS = REPO_ROOT / "jobs"

STAGE_DEFAULTS = {
    "build": {"touch_files": ["src/sources/b2mod_eirene.F", "src/modules/b2mod_eirene_training_dump.F90"],
              "expect_symbols": ["sheath_input", "surrogate_decision"], "make_jobs": 8},
    "campaign": {"save_points": "50 100 200 400 800 1600 3200 6400 12800 15000", "seeds": "1 2 3",
                 "dump_steps": 20, "amplitude": 0.2, "constant": 0.1,
                 "dropbox": "dropbox:SOLPS_DB/eirene_training_v3_transient__DIII-D__APP-FPP__uniform_dt1e-5"},
    "couple": {"nsteps": 50, "dt": "1.0e-5", "replay": 0, "dump": 0, "timeout": "300.0", "policy": "gate",
               "bundle": "", "service_warmup": 35, "overwrite": 0, "extra_b2mn": "", "case_group": "al_tests",
               "service_args": "", "keep_every": 10,
               "init_state": None},
    "train": {"commands": [], "submit_hint": "sbatch campaigns/nersc/train_eirene_sources.sbatch"},
}
# couple on perlmutter: the same keys plus the segment chain and the Slurm/launch pieces
STAGE_DEFAULTS["couple"].update({"nseg": 15, "seg_steps": 1000, "walltime": "12:00:00", "retrain_config": "", "retrain_interval": 600, "retrain_gpus": "1,2,3", "service_prefix": "",
                                 "start_state": None,
                                 "b2run_line": "b2run -m \\\"srun -n ${mpi_ranks} -c 2 --overlap --cpu-bind=cores\\\" b2mn >& run.log   # Perlmutter, validated 2026-10-15 (b2run is csh: the launcher words carry literal quotes)"})
EXT = {"build": "csh", "campaign": "sh", "couple": "sh", "train": "sh"}


def load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    for key in ("name", "stage", "host"):
        if key not in spec:
            raise ValueError(f"{path}: spec needs '{key}'")
    if spec["stage"] not in STAGE_DEFAULTS:
        raise ValueError(f"{path}: unknown stage {spec['stage']!r}; one of {sorted(STAGE_DEFAULTS)}")
    return spec


def load_host(name: str) -> dict:
    path = JOBS / "hosts" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no host file {path}")
    return yaml.safe_load(path.read_text())


def variables(spec: dict, host: dict) -> dict:
    """Host paths + stage defaults + spec params + derived names, all as strings."""
    v = dict(host)
    v.update(STAGE_DEFAULTS[spec["stage"]])
    v.update(spec.get("params") or {})
    v["name"], v["stage"] = spec["name"], spec["stage"]
    if "jobs_root" in v:
        v["log_dir"] = f"{v['jobs_root']}/logs"
        v["job_log"] = f"{v['log_dir']}/{spec['name']}.log"
    if spec["stage"] == "couple":
        if host.get("name") == "perlmutter" and v.get("start_state") and not v.get("init_state"):
            v["init_state"] = v["start_state"]
        if not v.get("init_state"):
            raise ValueError("couple needs params.init_state (a b2fstate/b2fstati on the host)")
        v.setdefault("start_state", v["init_state"])
        if "runs_root" in v:
            v["case_dir"] = f"{v['runs_root']}/{v['case_group']}/{spec['name']}"
        if int(v["replay"]) == 3 and v["policy"] != "none" and not v["bundle"]:
            raise ValueError("couple with replay 3 needs params.bundle (a bundle directory on the host)")
    if spec["stage"] == "train" and isinstance(v["commands"], list):
        v["commands"] = "\n".join(v["commands"])
    for key in ("touch_files", "expect_symbols"):
        if isinstance(v.get(key), list):
            v[key] = " ".join(v[key])
    for key in ("launch_env_tcsh", "launch_env_bash"):
        if key in v:
            v[key] = str(v[key]).rstrip("\n")
    if spec["stage"] == "couple":    # the service takes the policy as a flag
        v["policy_flag"] = {"always": "--always", "never": "--never"}.get(str(v.get("policy", "gate")), "")
    if "b2run_line" in v:          # the launch line may name the rank count
        v["b2run_line"] = str(v["b2run_line"]).replace("${mpi_ranks}", str(v.get("mpi_ranks", "")))
    return {k: "" if val is None else str(val) for k, val in v.items()}


class _Tpl(string.Template):
    idpattern = r"[a-z][_a-z0-9]*"     # driver variables are lower case; $D, $svc, $status stay


def render(spec_path: str | Path, out_dir: str | Path | None = None) -> Path:
    spec = load_spec(spec_path)
    host = load_host(spec["host"])
    tpl = JOBS / "templates" / f"{spec['stage']}.{spec['host']}.{EXT[spec['stage']]}"
    if not tpl.exists():
        raise FileNotFoundError(f"no template {tpl}")
    text = _Tpl(tpl.read_text()).safe_substitute(variables(spec, host))
    out = Path(out_dir or JOBS / "rendered") / f"{spec['name']}.{EXT[spec['stage']]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.chmod(0o755)
    return out


def _ssh(host: dict, command: str, timeout: int = 120) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host["ssh"], command],
                          capture_output=True, text=True, timeout=timeout + 30).stdout


def run(spec_path: str | Path) -> int:
    spec = load_spec(spec_path)
    host = load_host(spec["host"])
    script = render(spec_path)
    if "jobs_root" not in host:
        print(f"{host['name']}: rendered {script}; submit it there by hand (no job root on this host)")
        return 0
    v = variables(spec, host)
    remote = f"{v['jobs_root']}/{script.name}"
    subprocess.run(["ssh", "-o", "BatchMode=yes", host["ssh"], f"mkdir -p {v['log_dir']}"], check=True)
    subprocess.run(["scp", "-q", str(script), f"{host['ssh']}:{remote}"], check=True)
    launcher = "tcsh" if script.suffix == ".csh" else "bash"
    cmd = (f"cd {v['jobs_root']} && chmod +x {remote} && "
           f"(nohup setsid {launcher} {remote} > {v['job_log']} 2>&1 < /dev/null &) ; echo launched")
    print(_ssh(host, cmd).strip(), f"{spec['name']} on {host['name']}; log {v['job_log']}")
    return 0


def status(spec_path: str | Path, lines: int = 30) -> int:
    spec = load_spec(spec_path)
    host = load_host(spec["host"])
    v = variables(spec, host)
    if "job_log" not in v:
        print("no job log for this host")
        return 0
    print(_ssh(host, f"tail -n {int(lines)} {shlex.quote(v['job_log'])} 2>&1 | cut -c1-200"))
    return 0


def sync(host_name: str) -> int:
    """rsync the glue2 package and the SOLSTICE sources/configs to the host's glue root."""
    host = load_host(host_name)
    root = host["glue_root"]
    pairs = [(REPO_ROOT / "glue2", f"{root}/glue2"),
             (REPO_ROOT / "campaigns", f"{root}/campaigns"),
             (REPO_ROOT / "configs", f"{root}/configs"),
             (REPO_ROOT / "external/solstice/src", f"{root}/external/solstice/src"),
             (REPO_ROOT / "external/solstice/configs", f"{root}/external/solstice/configs")]
    subprocess.run(["ssh", "-o", "BatchMode=yes", host["ssh"], f"mkdir -p {root}/external/solstice"], check=True)
    for src, dst in pairs:
        subprocess.run(["rsync", "-aq", "--delete", "--exclude", "__pycache__", "--exclude", ".pytest_cache",
                        f"{src}/", f"{host['ssh']}:{dst}/"], check=True)
        print(f"synced {src.relative_to(REPO_ROOT)} -> {host['name']}:{dst}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("render", "run", "status"):
        p = sub.add_parser(name)
        p.add_argument("spec")
        if name == "status":
            p.add_argument("--lines", type=int, default=30)
        if name == "render":
            p.add_argument("--out", default=None)
    sub.add_parser("sync").add_argument("host")
    a = ap.parse_args(argv)
    if a.command == "render":
        print(render(a.spec, a.out))
        return 0
    if a.command == "run":
        return run(a.spec)
    if a.command == "status":
        return status(a.spec, a.lines)
    return sync(a.host)


if __name__ == "__main__":
    sys.exit(main())
