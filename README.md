# GLUE v2: active learning at the B2.5-EIRENE seam of SOLPS-ITER

GLUE v2 replaces the EIRENE Monte Carlo call inside SOLPS-ITER by a neural
surrogate (the SOLSTICE GNN) behind an uncertainty gate, and keeps the Monte
Carlo code as the teacher: every call the gate rejects is answered by EIRENE
and recorded as a training event. The cycle is predict, gate, fall back,
collect, retrain, version.

Lineage: the cycle and the error-bar/is-ok gate come from the LANL GLUE code
(C++ library plus Python service around LAMMPS); the plumbing here is new and
specific to SOLPS-ITER. The LAMMPS-era code is kept in the history at the tag
`glue1-lammps-archive`.

## Repository layout

```
glue2/          the package: events, catalog, ingest, snapshot, b2view (training view),
                learner (bundle contract), baseline, solstice_learner, loop, replay,
                surrogate_service, cli
configs/        loop configs (DIII-D EIRENE campaign: diiid_eirene_v3.*.yaml)
campaigns/      diiid_eirene: event generation on the SOLPS side (converged cases,
                transient trajectory, perturbed restarts, build and validation tools)
                nersc: training job (`glue2 train`) on Perlmutter
docs/           training-dump contract, data-flow diagram of what B2.5 receives from
                EIRENE, coupling note, hand-off notes
notebooks/      event-file explorers
tests/
legacy/         the SOLPEx 5-in/4-out socket prototype, reference only
external/       submodules: solstice (the GNN and its trainers, branch
                feature/eirene-sources) and the SOLPS-ITER fork (branch
                feature/eirene-training-dump) whose B2.5 writes the training events,
                replays them, and runs the surrogate mode
```

Data live outside the repository (`~/data/eirene_nn/{glue2_work,stores,runs}` on
the Mac, `$SCRATCH/eirene_nn` on Perlmutter, `/home/cloud/eirene_nn` on Mora).

## The three workflows

**Data.** The B2.5 fork, with `eirene_training_dump 1`, writes one NetCDF event
per EIRENE call: the plasma background B2.5 sent (BRAEIR), the raw return
(EIRBRA sources, wall tallies, estimators) and the call controls
(`docs/eirene_training_dump.md`). `glue2 ingest` catalogs event files,
`glue2 snapshot` freezes a content-addressed training set, `glue2.b2view`
trims it to what B2.5 feeds back: four sources per active stratum on the
interior mesh and the six wall tallies per stratum. A replay test inside
B2.5 proved this reduced set reproduces the plasma step exactly.

**Model.** `glue2 train` runs the SOLSTICE sources trainer on a snapshot
(`external/solstice`, `task: sources`): plasma fields as per-cell node
features, stratum strengths as FiLM parameters, one decoder head per source
kind, a pooled head for the per-stratum wall tallies. A seed ensemble is
wrapped by the `solstice_sources` learner into a bundle: mean, error bar
from the spread, novelty from the distance to the training set, and the
`ok` gate.

**Coupling.** B2.5's `eirene_training_replay 3` (surrogate mode) writes the
background before each EIRENE call and waits; `glue2.surrogate_service`
scores it with a bundle and answers with the prediction or with a fallback
marker. Fallback calls run EIRENE and are dumped as new events, so a coupled
run collects exactly the data it lacks. `glue2.replay` writes the files for
one-step replay tests; `campaigns/diiid_eirene/compare_b2fstate.py` compares
plasma states.

## Commands

```
glue2 ingest   --config configs/diiid_eirene_v3.mac.example.yaml   # new event files -> catalog
glue2 snapshot --config ...                                          # frozen training set
glue2 train    --config ... [--snapshot ID] [--name RUN] [--set k=v] # SOLSTICE sources trainer
glue2 cycle    --config ...                                          # the active-learning loop
python -m glue2.solstice_learner --snapshot S --mesh-store M --out-root R run1 run2 ...   # bundle
python -m glue2.replay --bundle B --template EVENT --out-dir D       # replay files for B2.5
python -m glue2.surrogate_service --bundle B --run-dir b2mn.exe.dir  # the live service
```

`pip install -e '.[dev]'` and `pytest` from the repository root; the trainer
needs `external/solstice` on `PYTHONPATH` (or installed) with torch and
torch_geometric.

## Work-directory layout

```
workdir/
  catalog.sqlite          events, rejected, requests, snapshots, bundles, audit
  snapshots/<id>/         data.nc (sample-stacked, float32) + manifest.json
  bundles/<id>/           bundle.json + member checkpoints or learner arrays
```

Splits are per case (`run_*` directory) from a salted hash, so all calls of a
run share a split and a run keeps its split as data are added. Cases that
originate from the loop's own requests (origin `al_request`, which is also
how transient trajectories are ingested) never enter `val`/`test`; their
held-out share is the `acq_test` split.

## Status (October 2026)

DIII-D, pure deuterium: 761 converged cases (one EIRENE call each) plus the
first transient events. Nine-seed bundle at or below EIRENE's own Monte Carlo
noise on the stratum-weighted sources of held-out converged cases. One-step
replay and live surrogate-mode runs on Mora reproduce the plasma step at the
noise level in the main SOL and within a few times the noise in the divertor
legs. The early transient of a uniform start is outside the training set and
is being added. Hand-offs with the details: `docs/nersc_handoff_eirene_sources.md`.
