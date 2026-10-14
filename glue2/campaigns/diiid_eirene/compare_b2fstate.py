#!/usr/bin/env python3
"""Compare two b2fstate files field by field (numpy only, runs on Mora).

    python3 compare_b2fstate.py A/b2fstate N/b2fstate [--rtol 1e-12]

Prints, per real field, the max |a - b| and max |a - b| / max |a|; the
label line (date and time) is ignored. Exit 0 when every field is within
rtol, 1 otherwise.
"""
import sys

import numpy as np


def read_b2fstate(path):
    fields, name, kind, buf = {}, None, None, []

    def flush():
        if name is not None and kind in ("real", "int"):
            fields[name] = np.array(buf, dtype=float if kind == "real" else int)

    with open(path) as f:
        for line in f:
            if line.startswith("*cf:"):
                flush()
                parts = line.split()
                kind, name, buf = parts[1], parts[3], []
            elif kind in ("real", "int"):
                buf.extend(line.split())
    flush()
    return fields


def main(argv):
    rtol = 1e-12
    if "--rtol" in argv:
        i = argv.index("--rtol")
        rtol = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    a, b = read_b2fstate(argv[0]), read_b2fstate(argv[1])
    bad = 0
    print(f"{'field':12s} {'n':>8s} {'max|a-b|':>12s} {'max|a-b|/max|a|':>16s}")
    for name in a:
        if name not in b or a[name].shape != b[name].shape:
            print(f"{name:12s} missing or shape mismatch")
            bad += 1
            continue
        d = np.abs(a[name] - b[name])
        scale = np.abs(a[name]).max()
        rel = d.max() / scale if scale > 0 else d.max()
        flag = "" if rel <= rtol else "  <-- differs"
        print(f"{name:12s} {a[name].size:8d} {d.max():12.4e} {rel:16.4e}{flag}")
        bad += rel > rtol
    print(f"{len(a) - bad} of {len(a)} fields within rtol {rtol:g}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
