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

Not in the template; to be run alongside it (a small script in `campaigns/nersc/` is the next thing
the Mac session writes, or write it here):

1. every N minutes: `glue2 ingest` of `$SCRATCH/eirene_nn/coupling/runs/gated/events/` with origin
   `al_request` (events of acquired calls never enter val/test), `glue2 snapshot`;
2. when `retrain_min_new` new events exist: fine-tune the current members (`train.init_from` each
   member, `init_strict false`, 100 epochs, lr 3e-4) on GPUs 1-3 (`CUDA_VISIBLE_DEVICES=1,2,3`),
   build the bundle with `--calib-splits val,acq_test`;
3. promote only if the converged `test` split does not regress by more than 5 %
   (`glue2.loop._promotion` has the rule); write the new bundle directory's path into
   `$SCRATCH/eirene_nn/coupling/runs/gated/reload_bundle`; the service swaps at its next call.

## What comes back to the Mac

`runs/<name>/states/state_NNNNN` (16 files, 9.5 MB each), `decisions.log`, `service.jsonl`,
`provenance` and the `events/` directory (training data; also upload to Dropbox
`SOLPS_DB/eirene_training_v3_transient__...` as cases). The Mac notebook compares the states with the
reference at each checkpoint against the Monte Carlo yardstick.
