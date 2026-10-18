"""Status file of a coupled run directory: ``status.json`` (machine) and ``STATUS.md`` (human).

    python -m glue2.runstatus start <run dir> key=value ...   # at launch: spec, bundle, start_state, policy, ...
    python -m glue2.runstatus refresh <run dir> [note=...] [verdict=...] [status=...]

``refresh`` derives what it can from the directory: segments (states/state_*), last step, start/end
times, the service log (requests, surrogate fraction, anchors, gate rejections, median service time),
events (fallback EIRENE calls), bundle(s) served, retrain cycles. Free-text ``note`` lines accumulate;
``verdict`` and ``status`` (running | complete | stopped | failed | superseded) are set explicitly.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

STATUS_JSON, STATUS_MD = "status.json", "STATUS.md"


def _load(run: Path) -> dict:
    p = run / STATUS_JSON
    return json.loads(p.read_text()) if p.exists() else {"name": run.name, "notes": []}


def _service(run: Path) -> dict:
    p = run / "service.jsonl"
    if not p.exists():
        return {}
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not recs:
        return {}
    sur = [r for r in recs if r.get("surrogate")]
    secs = [r["seconds"] for r in recs if "seconds" in r]
    return {"requests": len(recs), "surrogate_answers": len(sur), "surrogate_fraction": round(len(sur) / len(recs), 3),
            "anchors": sum(1 for r in recs if r.get("anchor")), "gate_rejected": sum(1 for r in recs if not r.get("ok", True)),
            "service_median_s": round(statistics.median(secs), 3) if secs else None,
            "bundles_served": sorted({r.get("bundle") for r in recs if r.get("bundle")}),
            "first_utc": recs[0].get("utc"), "last_utc": recs[-1].get("utc")}


def refresh(run: Path, **fields) -> dict:
    st = _load(run)
    states = sorted(glob.glob(str(run / "states" / "state_*")))
    st["segments_done"] = len(states)
    steps = [int(Path(x).name.split("_")[1]) for x in states if Path(x).name.split("_")[1].isdigit()]
    st["last_step"] = max(steps) if steps else None
    st["events"] = len(glob.glob(str(run / "events" / "*.nc"))) + len(glob.glob(str(run / "events" / "*" / "*.nc")))
    st["retrain_cycles"] = len(glob.glob(str(run / "retrain_cycle_*.log")))
    st["service"] = _service(run)
    for k, v in fields.items():
        if k == "note":
            st.setdefault("notes", []).append(f"{time.strftime('%Y-%m-%d %H:%M')} {v}")
        else:
            st[k] = v
    nseg = int(st.get("nseg", 15) or 15)
    if st.get("status") in (None, "running") and st["segments_done"] >= nseg:
        st["status"] = "complete"
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run / STATUS_JSON).write_text(json.dumps(st, indent=1))
    (run / STATUS_MD).write_text(_markdown(st))
    return st


def _markdown(st: dict) -> str:
    s = st.get("service") or {}
    lines = [f"# {st.get('name')}", "",
             f"**Status:** {st.get('status', 'unknown')}" + (f" — {st['verdict']}" if st.get("verdict") else ""), "",
             "| field | value |", "|---|---|"]
    for k in ("spec", "bundle", "model", "start_state", "policy", "service_args", "nseg", "seg_steps", "retrain_config", "host", "started", "job_ids"):
        if st.get(k) not in (None, ""):
            lines.append(f"| {k} | {st[k]} |")
    lines += [f"| segments done | {st.get('segments_done')} (last step {st.get('last_step')}) |",
              f"| fallback EIRENE events | {st.get('events')} |", f"| retrain cycles | {st.get('retrain_cycles')} |"]
    if s:
        lines += [f"| service requests | {s['requests']} (surrogate {s['surrogate_fraction']:.0%}, anchors {s['anchors']}, gate-rejected {s['gate_rejected']}) |",
                  f"| service median time | {s['service_median_s']} s |", f"| bundles served | {', '.join(s['bundles_served'])} |",
                  f"| service first / last | {s['first_utc']} / {s['last_utc']} |"]
    lines += ["", f"_updated {st.get('updated')}_", ""]
    if st.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in st["notes"]] + [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in ("start", "refresh"):
        print(__doc__)
        return 2
    run = Path(argv[1])
    fields = dict(a.split("=", 1) for a in argv[2:])
    if argv[0] == "start":
        fields.setdefault("status", "running")
        fields.setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    st = refresh(run, **fields)
    print(json.dumps({k: st.get(k) for k in ("name", "status", "segments_done", "last_step")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
