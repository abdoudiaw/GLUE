# GLUE v2 — active learning at the B2.5–EIRENE seam

GLUE v2 keeps the GLUE active-learning cycle — predict, gate on uncertainty,
request a fine-grain answer, accumulate ground truth, retrain, version — and
replaces its plumbing for SOLPS-ITER. The fine-grain answer is the EIRENE
return at `eirene_eirsrt` (schema-v2 and v3 training events); the model library is
SOLSTICE (through the `Learner` interface).

## What changed from GLUE v1, and why

GLUE v1 used SQLite as a message bus. Solver ranks inserted requests, the
service polled them, fine-grain jobs wrote another database that was merged
back, and retraining read the live tables. Several processes wrote one SQLite
file, often on a shared filesystem, which caused the persistent locking
failures.

| v1 | v2 |
|---|---|
| `BGKGND` rows hold the ground truth | Immutable event `.nc` files are the ground truth |
| Many writers, polling readers | One writer (lock file, fails fast); readers open read-only |
| Requests/results polled through SQLite | Nothing on the B2 iteration path; requests are rare, per cycle |
| Slurm-launched LAMMPS fine-grain jobs | A `Teacher` drops event files into an inbox |
| `retrain(dbHandle)` on live tables | Training reads a frozen, content-addressed snapshot |
| Learner swapped in during a run | Immutable bundles; promotion gated on held-out tests |
| Scalar SQL columns | Arrays stay in NetCDF; the catalog has one row per event |

## Repository layout

GLUE is the one checkout to work from. The simulator and the model library are
pinned as submodules of the GLUE repository (`git submodule update --init
external/solstice`; SOLPS-ITER is only needed for reading the source, it is
built on the run host from its own clone):

```
external/solstice     SOLSTICE (branch feature/eirene-sources): the GNN and its trainers
external/SOLPS-ITER   SOLPS-ITER fork (feature/eirene-training-dump) with the B2.5 fork
                      that writes the training events and carries the replay hook
glue2/          the package: events, catalog, ingest, snapshot, b2view, learner, loop
configs/        example loop configs (DIII-D EIRENE campaign: diiid_eirene_v3.*.yaml)
tests/
campaigns/      scripts that produce training events on the SOLPS side
  diiid_eirene/   DIII-D campaign runner (Mac/Mora controllers, per-case runner,
                  training-dump build, event validator)
  nersc/          job script: `glue2 train` of the sources model on Perlmutter
docs/           EIRENE training-dump contract, data-flow diagram of what B2.5
                receives from EIRENE, coupling note, hand-off notes
notebooks/      explorers for the event files (eirene_dump_explorer) and the
                legacy fort.31 background record (fort31_explorer)
legacy/         the SOLPEx 5-in/4-out socket prototype, kept for reference only
```

## Commands

```
python -m glue2.cli --config glue2.yaml ingest      # new event files -> catalog
python -m glue2.cli --config glue2.yaml snapshot    # frozen training set under workdir/snapshots
python -m glue2.cli --config glue2.yaml train [--snapshot ID] [--name RUN] [--device D] [--set k=v]
python -m glue2.cli --config glue2.yaml cycle       # the active-learning loop
```

`train` runs the SOLSTICE sources trainer (`external/solstice`, `task: sources`)
on the latest snapshot in a subprocess, using the `train:` block of the config
(training config, mesh store, output root). Data live outside the repository,
e.g. `~/data/eirene_nn/{glue2_work,stores,runs}`.

## Work-directory layout

```
workdir/
  catalog.sqlite          events, rejected, requests, snapshots, bundles, audit
  snapshots/<id>/         data.nc (sample-stacked, float32) + manifest.json
  bundles/<id>/           bundle.json + learner arrays
```

## The cycle (`glue2.loop.GlueLoop.cycle`)

1. **Ingest** new event files. A campaign case (with `eirene_training_v2.sha256`)
   is taken only after `EIRENE_TRAINING_SUCCESS` exists, and each file must match
   its manifest hash; the sampled controls in `source_params.json` go to the
   `cases` table. Other files must be older than `settle_seconds`, because the
   Fortran writer creates them in place. `_*` directories (quarantine) are never
   scanned, and cloud placeholders are left until they are downloaded. Invalid files go to
   `rejected` and are retried only if they change. Byte-identical copies are
   recorded once. An event whose background hash matches a request fulfils it.
2. **Retrain** once `retrain_min_new` new events exist. This freezes a snapshot,
   fits a candidate, and evaluates it on `test` (seed distribution) and
   `acq_test` (held-out acquired cases). The candidate is promoted only if
   neither split regresses beyond `promote_tolerance` and at least one improves.
3. **Acquire**: score unlabelled pool backgrounds with the promoted bundle and
   request the ones the gate rejects, most urgent first, up to
   `max_open_requests`.

Splits are per case (`run_*` directory), from a salted hash, so all repeats and
calls of one run share a split, and a run keeps its split as data are added.
Cases made only of acquired events never enter `val`/`test`.

## Learner contract (`glue2.learner`)

`fit(snapshot, out_root) -> bundle_dir`. A bundle's `predict(inputs)` returns
`mean` (native units), `errbar`, `score` (errbar / held-out RMSE), `novelty`
(>1 = outside training set) and `ok`. This is GLUE's `iserrok` for mesh fields.
`pca_ridge` is the reference baseline. A SOLSTICE learner registers its own
name with `glue2.learner.register`.

## Known data caveat (schema v2)

EIRENE index-maps `DELTA_SHEATH[XY]B` in place (`eirmod_infcop.F`) and B2 does not
refresh them before a repeated EIRENE call, so `braeir_delta_sheath[xy]` differ
between repeats of one background (the second has lost the ix=nx-1 target
entries). They are excluded from the background hash and should not be model
inputs for schema-v2 events.

Schema 3.0.1 events from a build with the B2 sheath save/restore compiled in
repeat these fields exactly, so they are valid inputs there (see
`configs/diiid_eirene_v3.example.yaml`). They remain outside the background hash
so that event identity does not depend on the build.

## Training on the result B2 used

On the first B2 call of a run EIRENE is called twice with identical inputs; B2
discards the first result. Schema-3 events record this in
`eirene_result_used_by_b2`, stored in the catalog as `used_by_b2`. Set
`used_only: true` in the snapshot spec to keep only the used result.

## Use

```bash
pip install -e '.[dev]'
glue2 status   --config configs/diiid_eirene_v2.example.yaml   # read-only, safe anytime
glue2 ingest   --config ...
glue2 snapshot --config ...
glue2 cycle    --config ... --cycles 10
pytest
```

`tests/test_loop.py::test_simulated_active_learning` runs the whole cycle. It
holds back part of an archive as the unlabelled pool and uses
`SimulatedTeacher` to answer requests from it. To run the same benchmark on
real data, point `pool.root` at held-back campaign cases.
