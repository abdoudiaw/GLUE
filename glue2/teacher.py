"""Candidate pools and teachers (the fine-grain side of the GLUE loop).

A pool offers plasma backgrounds that have no EIRENE answer in the catalog
yet; the loop scores them with the promoted bundle and requests the ones the
model does not trust. A teacher answers a request by producing event files
in an inbox directory, which the loop ingests like any other events. Teachers
never write the catalog.

* EventFilePool + SimulatedTeacher replay an existing archive: the pool shows
  only the BRAEIR inputs of held-back events, and the teacher "runs EIRENE" by
  releasing their files. This benchmarks active learning against real data.
* SpoolTeacher writes one JSON request per background for an external runner
  (e.g. the cloud campaign script launching a short SOLPS restart).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from glue2.events import EVENT_GLOB, EventError, read_arrays, read_event_info


@dataclass
class Candidate:
    candidate_id: str          # background hash
    case_id: str
    ref: str                   # where the background comes from (file path, fort.31, ...)


class Pool(Protocol):
    def candidates(self) -> list[Candidate]: ...

    def inputs(self, candidates: list[Candidate], names: list[str]) -> dict[str, np.ndarray]: ...


class Teacher(Protocol):
    def submit(self, requests: list[dict]) -> None: ...


@dataclass
class EventFilePool:
    """Backgrounds taken from event files; the first single_call event represents each."""

    root: Path
    files: dict[str, list[Path]] = field(default_factory=dict, init=False)
    _cands: dict[str, Candidate] = field(default_factory=dict, init=False)
    _cache: dict[tuple[str, str], np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        for path in sorted(self.root.rglob(EVENT_GLOB)):
            try:
                info = read_event_info(path)
            except EventError:
                continue
            self.files.setdefault(info.background_hash, []).append(path)
            if info.event_kind == "single_call" and info.background_hash not in self._cands:
                self._cands[info.background_hash] = Candidate(info.background_hash, info.case_id, str(path))

    def candidates(self) -> list[Candidate]:
        return list(self._cands.values())

    def inputs(self, candidates: list[Candidate], names: list[str]) -> dict[str, np.ndarray]:
        for cand in candidates:
            missing = [n for n in names if (cand.candidate_id, n) not in self._cache]
            if missing:
                for name, arr in read_arrays(cand.ref, missing).items():
                    self._cache[(cand.candidate_id, name)] = arr
        return {n: np.stack([self._cache[(c.candidate_id, n)] for c in candidates]) for n in names}


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    shutil.copy2(src, tmp)
    os.rename(tmp, dst)


@dataclass
class SimulatedTeacher:
    """Answers a request by copying every archived event of that background into the inbox."""

    pool: EventFilePool
    inbox: Path

    def submit(self, requests: list[dict]) -> None:
        for req in requests:
            for src in self.pool.files.get(req["background_hash"], []):
                _atomic_copy(src, Path(self.inbox) / req["case_id"] / src.name)


@dataclass
class SpoolTeacher:
    """Writes `<spool>/<request_id>.json` for an external runner; results return as event files."""

    spool: Path

    def submit(self, requests: list[dict]) -> None:
        spool = Path(self.spool)
        spool.mkdir(parents=True, exist_ok=True)
        for req in requests:
            tmp = spool / f".{req['request_id']}.json.tmp"
            tmp.write_text(json.dumps(req, indent=1, default=str))
            os.rename(tmp, spool / f"{req['request_id']}.json")
