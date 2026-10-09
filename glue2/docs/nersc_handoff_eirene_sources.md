# Hand-off: train the EIRENE sources model on NERSC

For whoever (or whatever) picks this up on Perlmutter. Written 2026-10-09 from the
Mac session that built the data set and the training code.

## What this is

We are replacing EIRENE calls inside SOLPS-ITER with a neural-network surrogate,
gated by GLUE2 (the GLUE active-learning loop rewritten for SOLPS). The surrogate
learns exactly what EIRENE returns to B2.5 at the `eirene_eirsrt` seam:

- **inputs**: the plasma background B2.5 hands to EIRENE (22 non-constant cell
  fields on the 36 x 96 interior B2 mesh: density, Te, Ti, velocities, fluxes,
  pressure, B field, sheath terms, potential, ...) plus the 7 stratum strengths
  B2.5 applies (`tflux * flux_scale`);
- **targets**: the raw per-stratum sources EIRENE returns — particle (`sni`),
  parallel momentum (`smo`), electron energy (`see`), ion energy (`sei`) for each
  of the 7 strata (`1W 2E 3S 4S 5N 6C 7V`) = 28 channels per cell. A replay test
  inside B2.5 proved these (plus six wall tallies, not trained yet) are all B2.5
  needs: feeding them back with everything else zeroed reproduces the plasma step
  bit for bit.

Data: 761 converged pure-deuterium DIII-D cases (schema-3 training events), one
EIRENE call each (the call B2 actually used), split 536 / 102 / 123 train / val /
test. Strata 1W and 2E (the two target plates) carry 97-99 % of the sources; the
loss weights reflect that but every channel is trained.

The model is the SOLSTICE GNN (encode-process-decode on a latent mesh), 1.29 M
parameters: plasma fields enter as per-cell node features, stratum strengths as
FiLM parameters, one decoder head per source kind.

Deadline context: APS-DPP talk in early November 2026 wants a trained surrogate
and a closed-loop coupling demo.

## Layout on NERSC

```
$SCRATCH/GLUE/                      git clone, branch glue2-solps-eirene (abdoudiaw/GLUE, private)
  glue2/glue2/                      the glue2 package (ingest, snapshot, b2view, cli ...)
  glue2/configs/diiid_eirene_v3.nersc.example.yaml   the config used here
  glue2/campaigns/nersc/train_eirene_sources.sbatch  the job script
  external/solstice/                submodule, branch feature/eirene-sources (abdoudiaw/solstice, private)
    configs/training/eirene_sources.yaml             the training recipe
    src/solstice/training/sources_data.py, sources_cli.py   the sources task
$SCRATCH/eirene_nn/
  glue2_work/diiid_eirene_v3/snapshots/43f5b89db3f5c301/{data.nc,manifest.json}  training set (458 MB)
  glue2_work/diiid_eirene_v3/catalog.sqlite                                      event catalog (read only here)
  stores/diiid_appfpp_eirene_v3_mesh.nc                                          mesh geometry (426 KB)
  runs/<name>/                                                                   training outputs
```

`$SCRATCH/nn/solstice` is a *different* checkout (branch `main`) used for the
COTSIM state models. Do not mix the two: nothing for this task is run or edited
there.

## How to run

```bash
cd $SCRATCH/GLUE && git pull && git submodule update --init external/solstice
ls $SCRATCH/eirene_nn/glue2_work/diiid_eirene_v3/snapshots/43f5b89db3f5c301/data.nc \
   $SCRATCH/eirene_nn/stores/diiid_appfpp_eirene_v3_mesh.nc        # both must exist

# smoke test (interactive GPU node or login node, ~2 min): proves paths and code
EPOCHS=2 NAME=smoke bash glue2/campaigns/nersc/train_eirene_sources.sbatch

# the real run: 1500 epochs, one A100, shared queue, resumes itself if resubmitted
sbatch glue2/campaigns/nersc/train_eirene_sources.sbatch
```

The script runs `python -m glue2.cli --config <config> train --name <NAME> ...`,
which launches `solstice.training.cli` (task `sources`) from `external/solstice`
on the latest snapshot in the catalog, with `PYTHONPATH` set to the glue2 package
and the solstice source. Slurm log: `logs/eirene/slurm-glue2-eirene-sources-<job>.out`.

Smoke-test success looks like:

```
snapshot /pscratch/.../43f5b89db3f5c301: 536 train / 102 val / 123 test; 3456 cells; inputs ['dni', 'vv', ...]; strata ['1W', '2E', '3S', '4S', '5N', '6C', '7V']
channel weights sni:1W=1.00 sni:2E=0.86 ... sei:2E=1.00 ...
gnn: 1,293,084 parameters
ep1: train 0.27 val 0.29 ...
== val (102 cases) ... == test (123 cases) ...
```

followed by `metrics.json` and `predictions.npz` in `$SCRATCH/eirene_nn/runs/smoke/`.
On the Mac GPU an epoch took 24 s; on an A100 expect a few seconds, so 1500
epochs should fit one 4 h slot. If it does not, resubmit the same command: it
resumes from `last.pt` (`train.max_hours` stops it cleanly before the slot ends).

Useful overrides (all go through `--set` of the training config):

```bash
NAME=eirene_sources_v3_s1 EXTRA="--set seed=1" sbatch glue2/campaigns/nersc/train_eirene_sources.sbatch
EPOCHS=300 NAME=short sbatch ...
```

A seed ensemble (seeds 0..4, same `data.split_seed`) is what GLUE2 will need
later for the uncertainty gate; one seed is enough for the first look.

## What to look at when it finishes

`$SCRATCH/eirene_nn/runs/<name>/metrics.json` reports, for val and test, in
native units:

- `weighted` per source kind: relative L2 of `sum_s W_s Y_s` (the field B2.5
  actually forms) per case, median and p90, and the relative error of its domain
  total. This is the number that matters for the coupling.
- `per_channel`: relative L2 per (source, stratum) channel.

Reference floor — the Monte Carlo noise between two EIRENE calls on the same
background (median over cases): sni ~12 %, see ~9 %, sei ~59 %, smo ~40 %. A
model at or below those numbers on the weighted fields is as good as EIRENE
itself; `sei` and `smo` will look worse than `sni`/`see` because of that noise,
not because of the network. For scale, the untrained smoke model gives ~45 %
(sni, see) and ~80 % (smo, sei).

`predictions.npz` holds `val_pred/val_true/test_pred/test_true` (case, cell,
channel) in native units, `val_W/test_W` (case, stratum), case ids, channel names
and `cell_perm` (cell -> flat iy*96+ix of the interior plane) for plotting.

Report back: the `metrics.json` numbers for val and test, the epoch/val-loss
curve (`log.jsonl`), wall time per epoch, and anything that failed.

## Things not to do

- Do not run training on anything but a GPU node, and do not train on the Mac.
- Do not commit or push with any Claude attribution (no `Co-Authored-By`, no
  "Generated with"): the author's git identity only.
- Do not touch `$SCRATCH/nn/solstice`, the solstice `main` branch, or the public
  `ORNL-Fusion/solstice` repository (its `main` is a force-pushed sanitized
  snapshot managed by `scripts/release/sync_public.sh`; never merge or push to it
  by hand).
- Do not edit the snapshot or the catalog; a new training set is made on the Mac
  with `glue2 ingest` / `glue2 snapshot` from the Dropbox archive of event files.
- Code changes for the sources task go on `external/solstice` branch
  `feature/eirene-sources` (then bump the submodule pointer in GLUE) or on GLUE's
  `glue2-solps-eirene`. Keep the state task (`task: state`, `training/data.py`,
  `cli.py` state path) unchanged; its tests must keep passing
  (`PYTHONPATH=src python -m pytest -q tests` in `external/solstice`; the 7
  `test_bundle` failures are a missing `safetensors` in some environments, not a
  regression).

## Where this goes next (not for this hand-off)

Wall-tally head (6 x 74 values per case), SOLSTICE learner registration with
GLUE2 (`errbar`/`ok` from the seed ensemble), and a surrogate mode in B2.5 (the
replay hook already in the B2.5 fork, extended to skip the EIRENE call) for the
closed-loop demo. The data-flow diagram of what B2.5 receives from EIRENE is in
`glue2/docs/eirene_return_dataflow.png`; the training-dump contract in
`glue2/docs/eirene_training_dump.md`.
