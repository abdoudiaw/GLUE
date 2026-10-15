# Hand-off: the coupled surrogate runs on Perlmutter

Written 2026-10-15 on the Mac for the NERSC session. Goal: runs 2 and 3 of the APS-DPP comparison
on one Perlmutter GPU node, with the retrain loop on the same node, so that no file has to travel
between machines during a run.

## The three runs

1. **Reference (done, on Mora)**: plain B2.5-EIRENE, training-dump build, from the step-50 state of a
   flat uniform start, 15 000 steps at dt 1e-5, as 15 chained 1000-step segments; plasma states every
   1000 steps. Lives on the Mac (`~/data/eirene_nn/reference_uniform_dt1e-5/`), where the comparison
   notebook runs. Its first and last states are in the package below (the last one for the build check).
2. **Surrogate only** (`jobs/specs/nersc_couple_surrogate_only.yaml`): the model answers every call
   after the first; no gate.
3. **Gated** (`jobs/specs/nersc_couple_gated.yaml`): surrogate where the gate accepts, EIRENE
   otherwise; every fallback call dumped as an event and retrained on. Anchor off by default
   (`--max-consecutive N` in `service_args` for the anchored variant as a separate run).

Both NN runs: same start state, same segment layout as the reference, states every 1000 steps,
copied back to the Mac for the notebook.

## What arrives from Mora (Dropbox `SOLPS_DB/coupling_package_20261015/`, 39 MB)

```bash
rclone copy dropbox:SOLPS_DB/coupling_package_20261015 $SCRATCH/eirene_nn/coupling/
cd $SCRATCH/eirene_nn/coupling && tar xzf template_chi1.tgz && tar xzf baserun.tgz
```

`template_chi1/` is the DIII-D case (symlinks resolved: AMJUEL, stra.dat, weis.dat are real files;
the `ds*` files are output scratch and may be deleted), `baserun/` the geometry b2run expects as
`../baserun`, `states/b2fstate_step00050` the common start, `states/ref_step15050` the hero's final
state, `reference_chain.sh` the Mora driver for reference.

## Build

- SOLPS-ITER fork `abdoudiaw/SOLPS-ITER`, branch `feature/eirene-training-dump`, commit `9777b64c`,
  with B2.5 submodule `abdoudiaw/B2.5` at `2fea8974b` (surrogate mode + contract-only request) and
  the Eirene submodule as pinned. Build target `b25eirene_mpi`; after any source change `touch` the
  file before `make` (make goes by mtime) and check `nm b2mn.exe | grep -c -i sheath_input` (3) and
  `surrogate_decision` (1).
- Host configuration: start from the ORNL gfortran one (`SETUP/setup.csh.ORNL.gfortran`) with
  cray-mpich and netcdf-fortran from `cray-hdf5`/`cray-netcdf`. Set `solps_tree` and
  `launch_env_bash` in `jobs/hosts/perlmutter.yaml` to what works, and the `b2run_line` default in
  `glue2/jobs.py` (or `params.b2run_line` in the specs) to the launch line of your build.
- B2.5 input switches used by the surrogate mode (all read from `b2mn.dat`):
  `eirene_training_replay 3`, `eirene_training_dump 1`, `eirene_surrogate_timeout 300.0`,
  `b2mwti_2dwrite 0`. Files exchanged in the directory b2mn runs in (`b2mn.exe.dir`):
  `eirene_surrogate_request.nc` (written by B2.5, ~1 MB), `eirene_surrogate_answer.nc` or
  `eirene_surrogate_fallback` (written by the service), `eirene_surrogate_decisions.log`.

## Build check (30 min, interactive or debug GPU job)

500 plain steps (`replay 0`, `dump 0`) from `states/ref_step15050`, then compare with the state the
Mora build reaches from the same point. The Mora reference drifts 0.5 % per 1000 steps in the
separatrix density and 1 to 2 % in the cell-wise norm (Monte Carlo scatter in the divertor legs); a
Perlmutter build of the same commit should stay within that. If it is off by more, stop and report.

## The job template (`jobs/templates/couple.perlmutter.sh`)

One node, 4 GPUs: the service on GPU 0 (`CUDA_VISIBLE_DEVICES=0`, `--device cuda`), B2.5 serial +
EIRENE on `mpi_ranks` cores, the case in node-local storage (`/dev/shm/eirene_surrogate/<name>`),
states/events/logs copied to `$SCRATCH/eirene_nn/coupling/runs/<name>/` at every segment end.
Resume with `sbatch --export=ALL,START_STATE=<last state>`. Render with
`python -m glue2.jobs render jobs/specs/<spec>.yaml` and submit the file in `jobs/rendered/`.

## The retrain loop for the gated run (GPUs 1-3, same node, separate process)

`campaigns/nersc/retrain_loop.sh configs/nersc_coupling.yaml` (copy the example
`configs/nersc_coupling.example.yaml`, set the bundle to warm-start from). One `glue2 cycle` every
`INTERVAL` seconds (default 600):

1. ingests `ingest_roots` as campaign events and `acquired_roots` (the run's `events/` directory
   and the Dropbox transient cases) with origin `al_request`, so those cases only ever land in
   `train` or `acq_test`;
2. when `retrain_min_new` (50) new eligible events exist, freezes a snapshot and fine-tunes every
   member of the `warm_start` bundle on it (`train.init_from` the member, `init_strict false`,
   `epochs` 100, `lr` 3e-4, `CUDA_VISIBLE_DEVICES=1,2,3` from the script), builds the bundle with
   `calib_splits: [val, acq_test]` and `fussiness` 1.75;
3. promotes it only if neither the converged `test` split nor `acq_test` regresses by more than
   `promote_tolerance` and one improves (`glue2.loop._promotion`), and then writes the bundle path to
   `reload_file`; the service (started with `--reload-file` on that path) swaps at its next call.

Stop it with a file named `STOP` next to the reload file. Everything it writes goes under the
GLUE2 workdir (`snapshots/`, `bundles/`, `bundles/.runs/<snapshot>/member<k>/`).

## What comes back to the Mac

`runs/<name>/states/state_NNNNN` (16 files, 9.5 MB each), `decisions.log`, `service.jsonl`,
`provenance` and the `events/` directory (training data; also upload to Dropbox
`SOLPS_DB/eirene_training_v3_transient__...` as cases). The Mac notebook compares the states with the
reference at each checkpoint against the Monte Carlo yardstick.

## Status 2026-10-15 12:00 (Perlmutter session)

Everything below ran on Perlmutter; the data are under `$SCRATCH/eirene_nn/coupling/` (NERSC) and
the code is on `main`.

**Build.** SOLPS-ITER fork 9777b64c (B2.5 2fea8974b, Eirene f8f63fa0 from jlore/EIRENE) built at
`/pscratch/sd/d/diawa/solps/solps-iter-eirene-training-dump`, host `NERSC.gfortran` (PrgEnv-gnu,
cray-mpich, cray-netcdf, libsci, MSCL built locally; empty ADAS placeholder; `b2mod_dimensions.mod`
copied into `modules/B2.5/src/modules/` for Eirene's cmake). Build check: 500 steps from the hero's
final state, compared with two Mora seeds over interior cells (`runs/build_check/comparison_interior.txt`):
Perlmutter-vs-Mora = density 1.9-2.1 %, velocity 4.5 %, Te 0.04 %, Ti 0.1 %, potential 0.9 %, ion flux
2.6-3.0 % against seed-to-seed 2.1 / 4.2 / 0.06 / 0.14 / 0.7 / 2.8 %. Same build within Monte Carlo.

**Model.** Snapshot `5057de7521c6cc55` (642 train incl. 106 transient events, val 102, test 123,
acq_test 20); 13 fine-tuned members; bundle `solstice_sources-2942df28b850` (acq_test gate 85 %,
rel_l2 0.24). Registered as promoted in the catalog; the online loop retrains from it.

**Run 2, surrogate on every call (`runs/surrogate_only`).** Segment 1 (steps 50-1050) completed at
1.5 s/step (`states/state_01050`); in segment 2 the (ignored) gate score rose from 0.8 to 300 and B2.5
aborted at step ~1787 with NaNs (`segments/run_02050.FAILED.log`). Result: without gate and retraining
the surrogate survives ~1700 steps of the transient, then diverges.

**Run 3, gated + online retraining (`runs/gated`).** Interactive route (`campaigns/nersc/couple_salloc.sh`,
driver on the Jupyter node, service and B2.5 as job steps, retrain loop beside the driver). Segment 1
(50-1050): ~100 % fallbacks, 4120 s, 1000 events dumped. Retrain cycles: #1 promoted 839792b6fac6
(+50 events; scores 3.2 -> 2.2), #2 promoted 41c6d56fe16d (1182 cases; acq_test rel_l2 0.239 -> 0.178;
scores -> 1.3-1.5, novelty 0.55-0.8). From then on the gate accepts: segment 2 (1050-2050) in 1718 s,
all surrogate, 1.33 s/step. Each dumped event is 13.6 MB (5.6 GB per 400 fallbacks); the retrain cycle
with 1182 cases takes ~40 min on one GPU (parallel members on GPUs 1-3 added in the learner,
`train_jobs/train_devices/train_prefix`).

**Operational findings.** Regular GPU queue: ~4950 pending, fairshare 0 -> interactive QOS (salloc only,
4 h, 2 jobs) is the practical route; premium is not enabled for m5186 (ask NERSC via the PI).
Steps inside a salloc inherit the enclosing Jupyter job's SLURM_* env (SLURM_MEM_PER_CPU etc.) and
then fail on 257 GB nodes: `couple_salloc.sh` unsets them. Running the driver itself as a job step
did not work reliably (abandoned). The service's 8 OMP threads cost EIRENE ~2 s/step (now 2).
`b2run -m \"srun -n 64 -c 2 --overlap --cpu-bind=cores\" b2mn` is the validated launch (literal quotes).
One driver per run: the script writes `$OUT/RUNNING` and exits if that job is alive.

**State at 12:00.** Run 3 at segment 3+, driver on Jupyter job 59808238 (expires ~12:20, kills the
driver); batch continuation job 59819617 (`jobs/rendered/sbatch/gated.sh`) queued in gpu regular.
Resume interactively: `nohup campaigns/nersc/couple_chunks.sh gated > coupling/runs/logs/chunks-gated.out 2>&1 &`
(cancel 59819617 first if it has not started).

**Next.** (1) finish run 3 (15 states); (2) run 2b: surrogate everywhere with run 3's final promoted
bundle; (3) run 4 per the NN-initial-state plan (Mac adds `b2fstati.nn_v2` and the nearest case
`run_1cd12aa5__D`'s final state to the package; 20-step restart check from `b2fstati.nn` already
passed); (4) states + logs of runs 2/2b/3 to the Mac for the comparison notebook; (5) delete
`runs/gated_try1`, `runs/surrogate_only_gatedbymistake`.
