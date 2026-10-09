"""Command line: python -m glue2 {status,ingest,snapshot,train,cycle} --config glue2.yaml

`status` opens the catalog read-only and can run while the loop is active.
`train` runs the model library's trainer on a snapshot (the latest by
default) in a subprocess, so GLUE itself never imports torch; it reads the
`train:` block of the config. The other commands take the catalog writer
lock and fail fast if it is held.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from glue2.catalog import Catalog, WriterBusy
from glue2.ingest import ingest
from glue2.loop import GlueLoop, LoopConfig
from glue2.snapshot import build_snapshot
from glue2.teacher import EventFilePool, SimulatedTeacher, SpoolTeacher

LOOP_KEYS = set(LoopConfig.__dataclass_fields__)


def load_config(path: str) -> tuple[LoopConfig, dict]:
    raw = yaml.safe_load(Path(path).read_text())
    return LoopConfig.from_dict({k: v for k, v in raw.items() if k in LOOP_KEYS}), raw


def make_pool_and_teacher(raw: dict):
    pool_cfg, teacher_cfg = raw.get("pool"), raw.get("teacher")
    pool = EventFilePool(Path(pool_cfg["root"])) if pool_cfg else None
    if not teacher_cfg:
        return pool, None
    if teacher_cfg["kind"] == "simulated":
        if pool is None:
            raise ValueError("simulated teacher needs an event-file pool")
        return pool, SimulatedTeacher(pool, Path(teacher_cfg["inbox"]))
    if teacher_cfg["kind"] == "spool":
        return pool, SpoolTeacher(Path(teacher_cfg["spool"]))
    raise ValueError(f"unknown teacher kind {teacher_cfg['kind']!r}")


REPO_ROOT = Path(__file__).resolve().parents[2]      # the GLUE checkout: glue2/glue2/cli.py


def latest_snapshot(cfg: LoopConfig) -> Path:
    """Newest snapshot of the catalog, located under the current workdir (the catalog
    stores the path it was built at, which changes when the workdir moves)."""
    with Catalog.reader(cfg.catalog_path) as cat:
        row = cat.query("SELECT snapshot_id FROM snapshots ORDER BY created_utc DESC LIMIT 1")
    if not row:
        raise SystemExit("glue2 train: no snapshot in the catalog; run `glue2 snapshot` first")
    return Path(cfg.workdir) / "snapshots" / row[0]["snapshot_id"]


def train(cfg: LoopConfig, raw: dict, snapshot: str | None, name: str | None,
          device: str | None, overrides: list[str]) -> int:
    """Train the model library's sources model on one snapshot.

    config `train:` block: `config` (the library's training config), `mesh_store`,
    `out_root` (default workdir/runs), `library_src` (default external/solstice/src
    of this checkout), `python` (default this interpreter). Relative paths resolve
    against the GLUE checkout."""
    t = raw.get("train") or {}
    if "config" not in t or "mesh_store" not in t:
        raise SystemExit("glue2 train: config needs train.config and train.mesh_store")
    snap = Path(snapshot) if snapshot and Path(snapshot).exists() else None
    if snap is None:
        snap = Path(cfg.workdir) / "snapshots" / snapshot if snapshot else latest_snapshot(cfg)
    if not (snap / "manifest.json").exists():
        raise SystemExit(f"glue2 train: {snap} is not a snapshot")
    lib_src = REPO_ROOT / t.get("library_src", "external/solstice/src")
    if not lib_src.exists():
        raise SystemExit(f"glue2 train: {lib_src} missing; run `git submodule update --init external/solstice`")
    out_root = Path(t.get("out_root") or Path(cfg.workdir) / "runs")
    out_dir = out_root / (name or f"{Path(t['config']).stem}_{snap.name}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(lib_src), str(REPO_ROOT / "glue2"), env.get("PYTHONPATH", "")])
    cmd = [t.get("python") or sys.executable, "-m", "solstice.training.cli",
           "--config", str(REPO_ROOT / t["config"]),
           "--set", f"data.snapshot={snap}", "--set", f"data.mesh_store={REPO_ROOT / t['mesh_store']}",
           "--set", f"out_dir={out_dir}"]
    for item in overrides:
        cmd += ["--set", item]
    if device:
        cmd += ["--device", device]
    print("glue2 train:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env, cwd=REPO_ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="glue2")
    ap.add_argument("command", choices=["status", "ingest", "snapshot", "train", "cycle"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--max-new-cases", type=int, help="override max_new_cases for this run")
    ap.add_argument("--snapshot", help="train: snapshot id or path (default: latest)")
    ap.add_argument("--name", help="train: run name under train.out_root")
    ap.add_argument("--device", help="train: cuda | cpu | mps")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="train: override an entry of the training config (e.g. train.epochs=2)")
    args = ap.parse_args(argv)
    cfg, raw = load_config(args.config)
    if args.max_new_cases is not None:
        cfg.max_new_cases = args.max_new_cases

    if args.command == "status":
        with Catalog.reader(cfg.catalog_path) as cat:
            print(json.dumps(cat.summary(), indent=1))
        return 0
    if args.command == "train":
        return train(cfg, raw, args.snapshot, args.name, args.device, args.set)

    try:
        cat = Catalog.writer(cfg.catalog_path)
    except WriterBusy as exc:
        print(f"glue2: {exc}", file=sys.stderr)
        return 2
    with cat:
        if args.command == "ingest":
            rep = ingest(cat, cfg.ingest_roots, settle_seconds=cfg.settle_seconds,
                         exclude=cfg.exclude, max_new_cases=cfg.max_new_cases,
                         skip_dataless=cfg.skip_dataless)
            print(json.dumps(rep.as_dict()))
        elif args.command == "snapshot":
            snap = build_snapshot(cat, cfg.spec, Path(cfg.workdir) / "snapshots")
            print(json.dumps({"snapshot_id": snap.snapshot_id, "counts": snap.manifest["counts"]}))
        else:
            pool, teacher = make_pool_and_teacher(raw)
            loop = GlueLoop(cfg, cat, pool, teacher)
            for _ in range(args.cycles):
                print(json.dumps(loop.cycle(), default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
