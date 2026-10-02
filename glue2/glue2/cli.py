"""Command line: python -m glue2 {status,ingest,snapshot,cycle} --config glue2.yaml

`status` opens the catalog read-only and can run while the loop is active.
The other commands take the catalog writer lock and fail fast if it is held.
"""

from __future__ import annotations

import argparse
import json
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="glue2")
    ap.add_argument("command", choices=["status", "ingest", "snapshot", "cycle"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--max-new-cases", type=int, help="override max_new_cases for this run")
    args = ap.parse_args(argv)
    cfg, raw = load_config(args.config)
    if args.max_new_cases is not None:
        cfg.max_new_cases = args.max_new_cases

    if args.command == "status":
        with Catalog.reader(cfg.catalog_path) as cat:
            print(json.dumps(cat.summary(), indent=1))
        return 0

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
