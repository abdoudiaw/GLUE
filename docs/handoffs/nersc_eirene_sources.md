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
$SCRATCH/GLUE/                      git clone, branch main (abdoudiaw/GLUE, private)
  glue2/                      the glue2 package (ingest, snapshot, b2view, cli ...)
  configs/diiid_eirene_v3.nersc.example.yaml   the config used here
  campaigns/nersc/train_eirene_sources.sbatch  the job script
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
EPOCHS=2 NAME=smoke bash campaigns/nersc/train_eirene_sources.sbatch

# the real run: 1500 epochs, one A100, shared queue, resumes itself if resubmitted
sbatch campaigns/nersc/train_eirene_sources.sbatch
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
NAME=eirene_sources_v3_s1 EXTRA="--set seed=1" sbatch campaigns/nersc/train_eirene_sources.sbatch
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
  `main`. Keep the state task (`task: state`, `training/data.py`,
  `cli.py` state path) unchanged; its tests must keep passing
  (`PYTHONPATH=src python -m pytest -q tests` in `external/solstice`; the 7
  `test_bundle` failures are a missing `safetensors` in some environments, not a
  regression).

## Status 2026-10-09 (Perlmutter)

Smoke test passed; four seeds (0-3) trained on a Jupyter node in 28 min each
(1.1 s/epoch, best validation epoch 102-126 of 1500, i.e. the schedule can be cut
~10x). Test, weighted fields, rel L2 median: sni 10.5-11.9 %, see 10.6-11.8 %,
smo 23.9-27.3 %, sei 33-35 %: at or below the EIRENE Monte Carlo floor on all
four. Runs: `$SCRATCH/eirene_nn/runs/eirene_sources_v3_jnode_s{0..3}`, summary
`$SCRATCH/eirene_nn/runs/SUMMARY.md`. Batch jobs 59601584 (seed 0) and
59601657/59/60/61 (seeds 1-4) queued for the weekend under
`runs/eirene_sources_v3{,_s1..4}`.

Update 2026-10-14: the five batch jobs completed the same evening (30 min each),
so nine seeds exist; `runs/SUMMARY.md` (and `runs/summarize.sh`) now cover all
nine. Test, weighted fields, rel L2 median over the nine: sni 10.5-12.7 %,
see 10.5-11.8 %, smo 23-28 %, sei 33-36 %; best validation epoch 93-126 for every
seed. Nine-seed bundle: `runs/bundles/solstice_sources-cfc800459d3a` (gate accepts
84 % of val and 89 % of test at fussiness 1.5; error-bar scales 5.2-11, so nine
seeds on one split are still under-dispersed and bootstrap splits remain the next
lever for the error bar). Decided: grow the training set, in particular with
early-transient-phase EIRENE calls, only after the B2.5 coupling works.

### GLUE2 learner: `solstice_sources` (`glue2/solstice_learner.py`)

The SOLSTICE seed ensemble behind the Learner/Bundle protocol (mean, errbar,
score, novelty, ok), registered next to `pca_ridge` (`learner: solstice_sources`
in the config; `learner_params`: `members`, `epochs`, `mesh_store`, `fussiness`,
`novelty_max`, `knn`, or `runs: [...]` to wrap trained runs). `fit()` trains the
seeds with the same command as `glue2 train`, then calibrates on the validation
split: errbar per source kind = member spread in the trainer's normalized metric
x a scale that matches the mean error; `rmse` = held-out error in that metric;
`score = max errbar / rmse`, gate `score <= fussiness` and `novelty <= novelty_max`
(k-NN distance of a per-case summary of the normalized inputs to the training
cases). `predict()` returns the sources in the event layout, so
`glue2.learner.evaluate` works unchanged.

Build a bundle from existing runs and score it:

```bash
cd $SCRATCH/GLUE && export PYTHONPATH=$PWD:$PWD/external/solstice/src
python -m glue2.solstice_learner --snapshot $SCRATCH/eirene_nn/glue2_work/diiid_eirene_v3/snapshots/43f5b89db3f5c301 \
  --mesh-store $SCRATCH/eirene_nn/stores/diiid_appfpp_eirene_v3_mesh.nc --out-root $SCRATCH/eirene_nn/runs/bundles \
  $SCRATCH/eirene_nn/runs/eirene_sources_v3_jnode_s{0,1,2,3}
```

First bundle (4 seeds): `runs/bundles/solstice_sources-67aa0a32036a`; the gate
accepts 87 % of val and 89 % of test cases at fussiness 1.5. Error-bar scales are
6-12 (four seeds on one split under-disperse), so more seeds or bootstrap splits
would sharpen the error bar. Note that `evaluate()` scores the raw per-stratum
arrays (unweighted, all strata), so its `smo` numbers look poor (signed field,
domain sum near zero); the stratum-weighted metrics of `metrics.json` are the
ones that matter for B2.5.

## Status 2026-10-14 (Mac): transient data and the tally head

The first B2.5 coupling test ran on Mora (one-step replay of the 9-seed surrogate on
the evolved transient case): mechanics work, the masked-truth step is identical to
the true step (momentum slots 1-2 and unused strata do not matter), the surrogate
step is at the Monte Carlo noise level in the main SOL and 3-4x above it in the
divertor legs. Scoring the early uniform-start transient (steps 50-120 of the
`acceleration_tests` trajectory) with the bundle gives gate scores of 11-12 and
novelty 2.3: that regime is absent from the training set. 84 transient events
(4 cases, `run_tr00050_s{0,1}__D`, `run_tr00100_s{0,1}__D`; `_s1` are perturbed
restarts) were ingested on the Mac with origin `al_request`, so they never enter
the converged val/test splits:

    snapshot 48502aa2f61dc686: train 596 (536 converged + 60 transient), val 102,
    test 123 (converged, unchanged), acq_test 20 (one transient case, held out)

The snapshot directory (497 MB) is copied by the user into
`$SCRATCH/eirene_nn/glue2_work/diiid_eirene_v3/snapshots/48502aa2f61dc686/`.

GLUE commit `082a267` / SOLSTICE `809d988` (pushed) add the wall-tally head
(`model.tally_head`, 6 tallies x 7 strata x 74 surfaces as a pooled global output,
on by default in `eirene_sources.yaml`) and `train.init_strict: false`, which lets
a checkpoint without the head initialise a model with it. `b2view.py` now returns
the tallies per stratum, which is how B2.5 consumes them; the old snapshot stores
all stratum slots, so no snapshot rebuild is needed for that.

### Recipe: fine-tune the existing seeds on the new snapshot, with the tally head

One run per existing seed run (the nine from `runs/SUMMARY.md`, plus seeds 5-8 when
they exist), all parameters trainable, lower learning rate, short schedule. Verified
on the Mac at 2 epochs (`init_strict` reports the six `global_head.*` tensors as
fresh); on an A100 expect ~1.5 s/epoch, so ~5 minutes per seed:

```bash
cd $SCRATCH/GLUE && git pull && git submodule update --init external/solstice
export PYTHONPATH=$PWD:$PWD/external/solstice/src
for s in 0 1 2 3 4 5 6 7 8; do
  SRC=$SCRATCH/eirene_nn/runs/<run dir of seed s>/best.pt
  python -m glue2.cli --config configs/diiid_eirene_v3.nersc.example.yaml train \
     --name eirene_sources_v3t_s$s --snapshot 48502aa2f61dc686 \
     --set seed=$s --set train.init_from=$SRC --set train.init_strict=false \
     --set train.epochs=200 --set train.lr=3.0e-4 --set train.warmup=5
done
python -m glue2.solstice_learner --snapshot $SCRATCH/eirene_nn/glue2_work/diiid_eirene_v3/snapshots/48502aa2f61dc686 \
  --mesh-store $SCRATCH/eirene_nn/stores/diiid_appfpp_eirene_v3_mesh.nc --out-root $SCRATCH/eirene_nn/runs/bundles \
  $SCRATCH/eirene_nn/runs/eirene_sources_v3t_s{0..8}
```

What to report: (1) `metrics.json` of one run: the weighted-field numbers on `test`
must not regress against the 9-seed bundle (sni ~11 %, see ~11 %, smo ~25 %,
sei ~34 %), and the new `tallies` block (per-tally rel L2 and per-stratum surface
sums); (2) the bundle's `evaluate` on `acq_test` (the held-out transient case):
gate acceptance and rel_l2 there are the point of the exercise, they were 0 % and
>100 % before; (3) the bundle id and its calibration. The bundle then goes back to
the Mac (tar + JupyterLab download, as before) for the coupled run on Mora.

## Where this goes next (not for this hand-off)

Wall-tally head (6 x 74 values per case), SOLSTICE learner registration with
GLUE2 (`errbar`/`ok` from the seed ensemble), and a surrogate mode in B2.5 (the
replay hook already in the B2.5 fork, extended to skip the EIRENE call) for the
closed-loop demo. The data-flow diagram of what B2.5 receives from EIRENE is in
`docs/eirene_return_dataflow.png`; the training-dump contract in
`docs/eirene_training_dump.md`.
