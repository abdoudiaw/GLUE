# GLUE v2: Active learning for SOLPS-ITER–EIRENE

GLUE v2 uses the SOLSTICE GNN to replace SOLPS-ITER’s EIRENE calls when its predictions pass an uncertainty gate. Otherwise, EIRENE runs and its results are added to the training data.

The active-learning cycle and gate follow LANL’s GLUE for LAMMPS. The SOLPS-ITER integration is new.

## Repository layout

```text
glue2/          Data pipeline, training, active-learning loop, replay, service
configs/        Loop configurations
campaigns/
  diiid_eirene/ Event generation and validation
  nersc/        Perlmutter training jobs
docs/           Data contract, coupling notes, handoffs
notebooks/      Event-file exploration
tests/          Tests
legacy/         Reference socket prototype
jobs/           Build, campaign, training, and coupling job specs
external/       SOLSTICE and SOLPS-ITER forks
```

Data are stored outside the repository:

- Mac: `~/data/eirene_nn/{glue2_work,stores,runs}`
- Perlmutter: `$SCRATCH/eirene_nn`
- Mora: `/home/cloud/eirene_nn`

## Workflows

### Data

With `eirene_training_dump 1`, the SOLPS-ITER fork writes one NetCDF event per EIRENE call. Each event contains the B2.5 input, EIRENE outputs, and call controls. See `docs/eirene_training_dump.md`.

`glue2 ingest` catalogs events; `glue2 snapshot` freezes a training set. `glue2.b2view` keeps the fields B2.5 needs. A replay test confirmed that this reduced set reproduces the plasma step.

### Training

`glue2 train` runs the SOLSTICE sources trainer on a snapshot. The `solstice_sources` learner packages an ensemble with predictions, uncertainty, novelty, and a gate decision.

### Coupling

In surrogate mode, B2.5 sends each background to `glue2.surrogate_service`. The service returns a prediction or requests an EIRENE fallback. Fallback results are saved as new training events.

`glue2.replay` prepares files for one-step tests. `campaigns/diiid_eirene/compare_b2fstate.py` compares plasma states.

## Commands

```sh
glue2 ingest   --config configs/diiid_eirene_v3.mac.example.yaml
glue2 snapshot --config <config.yaml>
glue2 train    --config <config.yaml> [--snapshot ID] [--name RUN] [--set k=v]
glue2 cycle    --config <config.yaml>

python -m glue2.solstice_learner --snapshot S --mesh-store M --out-root R run1 run2 ...
python -m glue2.replay --bundle B --template EVENT --out-dir D
python -m glue2.surrogate_service --bundle B --run-dir b2mn.exe.dir

python -m glue2.jobs render jobs/specs/<spec>.yaml
python -m glue2.jobs run    jobs/specs/<spec>.yaml
python -m glue2.jobs status jobs/specs/<spec>.yaml
python -m glue2.jobs sync   mora
```

Install and test from the repository root:

```sh
pip install -e '.[dev]'
pytest
```

Training requires `external/solstice` on `PYTHONPATH` or installed, with PyTorch and PyTorch Geometric.

## Work directory

```text
workdir/
  catalog.sqlite
  snapshots/<id>/  data.nc, manifest.json
  bundles/<id>/    bundle.json, checkpoints or learner arrays
```

Splits are assigned by case (`run_*`) using a salted hash. Calls from one run stay together, and assignments persist as data grow. Loop-requested cases (`al_request`) are excluded from `val` and `test`; held-out cases use `acq_test`.

## Status (October 2026)

DIII-D pure-deuterium data include 761 converged cases and initial transient events. A nine-seed bundle is within EIRENE’s Monte Carlo noise on held-out converged cases. One-step replay and live surrogate runs on Mora reproduce plasma steps near the noise level in the main SOL and within a few times the noise in the divertor legs.

The early transient from a uniform start is being added. See `docs/handoffs/nersc_eirene_sources.md`.
