# SOLPS–EIRENE Training Dump: Test Handoff

**Project:** SOLPS-ITER–SOLSTICE neutral-source coupling  
**Project owner:** Abdou Diaw  
**Date:** 2026-10-06

## Bottom line

The schema-3.0.1 training dump is working and has passed its strict validation test. It records the complete intended B2→EIRENE input snapshot and the raw EIRENE→B2 return contract for this fixed DIII-D configuration.

The unresolved problem is separate: the current modified SOLPS executable writes the valid EIRENE events and then B2 stops with `Supra-luminal velocities` in `b2nph9`. The older SOLPS executable advances the identical flat initial state successfully. Tests have ruled out the NetCDF logger, `eirene_repeat_first_call`, and the new sheath-restoration commit as the cause.

Do not restart the 761-case campaign yet. First isolate the old-versus-new executable regression or use a restart state that the current executable can advance.

## Repositories and pushed revisions

SOLPS fork and branch:

- Repository: <https://github.com/abdoudiaw/SOLPS-ITER>
- Branch: `feature/eirene-training-dump`
- Current SOLPS-ITER commit: `5f9a93bf` — `Store active B2 mesh in EIRENE training inputs`

B2.5 fork and branch:

- Repository: <https://github.com/abdoudiaw/B2.5>
- Branch: `feature/eirene-training-dump`
- Current B2.5 commit: `45d122739` — `Store active B2 mesh in EIRENE training inputs`

Relevant B2.5 history:

- `f03d24aff` — capture complete EIRENE return state (schema v2)
- `4be4cac89` — add training-dump validator
- `fe9b1a55c` — capture BRAEIR immediately before `eirene_eirsrt`
- `3fc644daa` — restore sheath inputs around repeated EIRENE calls
- `76feca666` — capture the EIRENE index map after its first initialization
- `45d122739` — store only the active pre-map B2 mesh in BRAEIR inputs

The parent SOLPS commits point to the matching B2.5 commits. Both repositories were pushed. The ORNL-Fusion `solstice` repository was not changed.

## What schema 3.0.1 fixes

The DIII-D case has:

- B2 interior mesh: `nx=96`, `ny=36`
- B2 mesh including guard cells: `98 × 38`
- EIRENE-indexed mesh after two inserted cut columns: `100 × 38`

The first schema-v3 implementation copied the entire 100-column BRAEIR allocation before EIRENE. Columns 98–99 are scratch space before the in-place EIRENE index mapping. On a repeated call, those two inactive columns retained values from the preceding mapping, making ten input arrays appear different at exactly 76 entries (`2 × 38`).

This was a logger-storage problem, not a different plasma background. Schema 3.0.1 now writes:

- `bra_x=98`, `bra_y=38` for pre-map BRAEIR inputs;
- `eir_x=100`, `eir_y=38` for raw EIRBRA outputs;
- `b2_mesh_nx_interior=96` and `b2_mesh_ny_interior=36`;
- the EIRENE index map (`NCUTB`, `NCUTL`, `TARGINDEX`, and `NPOINT`);
- `braeir_storage_policy="active B2 mesh including guard cells; EIRENE scratch excluded"`.

For this mesh, `NPOINT` is:

```text
[[ 1, 25],
 [26, 74],
 [75, 99]]
```

The two repeats now have bit-identical values for all 42 recorded B2→EIRENE input fields. Their EIRENE outputs differ slightly because the Monte Carlo calls are independent; that is expected.

## Validated reference output

Run directory:

```text
/home/cloud/solps-runs/diii-d/acceleration_tests/uniform_dt1e-5_seedfix_v3_smoke
```

The latest validated logger-on pair was moved to:

```text
/home/cloud/solps-runs/diii-d/acceleration_tests/uniform_dt1e-5_seedfix_v3_smoke/rerun_20261006_logger_on_validated
```

Files:

```text
eirene_training_v3_b2call_00000000_single_call_0001.nc
eirene_training_v3_b2call_00000000_single_call_0002.nc
```

Metadata:

```text
schema_version = "3.0.1"
b2_5_git       = "45d1227"
solps_iter_git = "5f9a93b"
```

Repeat semantics:

- repeat 1/2: `eirene_result_used_by_b2=0` (warm-up result not used by B2);
- repeat 2/2: `eirene_result_used_by_b2=1` (result used by B2).

Validation result:

```text
PASS B2 call 0: 2 repeats have bit-identical inputs (42 fields)
PASS validated 2 EIRENE training event(s)
```

Validation command:

```bash
python3 /home/cloud/local/solps/solps-iter-eirene-training-dump/modules/B2.5/src/test/validate_eirene_training_dump.py /home/cloud/solps-runs/diii-d/acceleration_tests/uniform_dt1e-5_seedfix_v3_smoke/rerun_20261006_logger_on_validated
```

A valid single-call event from the `eirene_repeat_first_call=0` A/B test is in:

```text
/home/cloud/solps-runs/diii-d/acceleration_tests/uniform_dt1e-5_seedfix_v3_smoke/repeat_first_call_0_validated
```

It has `repeat=1/1` and `eirene_result_used_by_b2=1` and passes the validator.

Older development outputs are preserved under `pre_map_fix_dumps`, `pre_active_mesh_fix_dumps`, and `pre_provenance_fix_dumps`. Do not use those for new training.

## Uniform-state failure and completed A/B tests

Initial state:

```text
/home/cloud/solps-runs/diii-d/acceleration_tests/uniform_dt1e-5_seedfix/b2fstati
```

The smoke copy has the identical `b2fstati`:

```text
SHA256 c981788044c8da63e84e54c75dcab74ebdf35e9213947e512277029ad54409cc
```

The smoke controls are restored to:

```text
b2mndr_ntim=1
b2mndr_dtim=1.0e-5
eirene_training_dump=1
eirene_repeat_first_call=1
```

Every failing current-build run reaches EIRENE, writes the requested event file(s), and then stops while B2 advances:

```text
*** XERRAB: program will stop. ***
Supra-luminal velocities !
Call chain follows.
 b2nph9
 b2news_
 b2mndt
 b2mndr_1
 b2mn
```

Important: `b2run`/the runs Makefile ignores the `b2mn` error, so the shell command may return status zero. Detect failure from the log and from the truncated `b2fstate` (777 bytes), not only the shell status.

Completed test matrix:

| Executable/configuration | EIRENE dump | B2 result |
|---|---:|---|
| Current build, logger on, `repeat_first_call=1` | Two valid events | Supra-luminal stop |
| Current build, logger off, `repeat_first_call=1` | None, as expected | Same supra-luminal stop |
| Current build, logger on, `repeat_first_call=0` | One valid event | Same supra-luminal stop |
| Current build with commit `3fc644daa` temporarily removed, logger off | None | Same supra-luminal stop |
| Old `/home/cloud/local/solps/solps-iter-3.0.8-devel` executable, identical `b2fstati`, one step | No new logger support | Success; normal 11 MB `b2fstate` |

Conclusions:

1. The NetCDF logger is observational and does not cause the failure.
2. The repeated warm-up EIRENE call is not required for the failure.
3. The sheath save/restore commit is not the cause.
4. The same initial state is accepted by the old executable.
5. The remaining regression lies elsewhere between the old build/source and the current Jeremy-based fork/build.

The original successful flat-state run completed 20,000 iterations and has only the initial and final restart files; no intermediate `b2fstate.XXXX` files remain.

## Current Mora source/build state

Source tree:

```text
/home/cloud/local/solps/solps-iter-eirene-training-dump
```

Expected revisions:

```text
SOLPS-ITER  5f9a93b
B2.5       45d1227
```

The temporary no-sheath source was removed. The pushed production source was restored and the production MPI executable was rebuilt successfully.

Production executable:

```text
/home/cloud/local/solps/solps-iter-eirene-training-dump/modules/B2.5/builds/couple_SOLPS-ITER.ORNL.gfortran.mpi/b2mn.exe
```

The repository contains untracked build logs, but no intended tracked source differences should remain. Verify before further work:

```bash
cd /home/cloud/local/solps/solps-iter-eirene-training-dump
git status --short
git -C modules/B2.5 status --short
git rev-parse --short HEAD
git -C modules/B2.5 rev-parse --short HEAD
```

## Build and runtime environment on Mora

Run setup from the SOLPS repository directory; sourcing `setup.csh` from the run directory fails because it resolves configuration relative to the current directory.

```tcsh
cd /home/cloud/local/solps/solps-iter-eirene-training-dump
setenv SOLPS_HOST_NAME_FORCE ORNL
source setup.csh gfortran
setenv GLI_HOME /home/cloud/local/solps/solps-libs
setenv MSCL_ROOT /home/cloud/local/solps/solps-libs/lib
setenv HWLOC_COMPONENTS "-gl"
setenv LD_LIBRARY_PATH /home/cloud/local/solps/solps-libs/lib:${LD_LIBRARY_PATH}
```

`HWLOC_COMPONENTS=-gl` avoids the OpenMPI/hwloc GL component hanging while trying to connect to the display. `LD_LIBRARY_PATH` is required for `libmscl.so.1`.

Build command:

```tcsh
make -j8 b25eirene_mpi > build.log &
```

Run command from a case directory:

```tcsh
b2run -m \"mpirun --bind-to none -x HWLOC_COMPONENTS -x LD_LIBRARY_PATH -np 8\" b2mn > run.log &
```

Do not use `tee` for these runs. Use one command line and background it when working interactively.

## Recommended next steps

1. **Verify the restored production tree and executable.** Confirm the two Git revisions above and no tracked differences.
2. **Test the current executable from the old successful final restart.** Make a new work directory and use the old run's 11 MB `b2fstate` as `b2fstati`. Do not overwrite the original run. This determines whether the regression is limited to the perfectly flat state.
3. **Compare or bisect old and new B2 solver code.** The old checkout is a monolithic tree at parent revision `2bdde7a3`; `modules/B2.5` is not represented as the same submodule history, so a naive B2 submodule hash comparison does not work.
4. **Instrument `b2nph9` diagnostically.** Before changing numerical behavior, print the offending cell, species, velocity, density, and limiting velocity. Compare old and new builds at the first B2 step.
5. **Compare build flags and relevant numerical routines.** Focus on `b2nph9`, `b2news_`, state reading, boundary initialization, and any changes between the old monolithic B2 source and the current B2.5 fork.
6. **Keep the logger enabled during diagnostics only when dumps are needed.** Logger-off already proved that the dump code does not cause the failure.
7. **Do not regenerate the large campaign yet.** The v3.0.1 contract is ready, but trajectory generation from a flat state needs a B2 configuration/current executable that advances successfully.

If the immediate goal is model work rather than resolving the solver regression, use the validated schema-3.0.1 pair above and the existing converged-case data. Treat the old run's final `b2fstate` as an evolved restart, not as a flat initial state.
