# Documentation

- `eirene_training_dump.md`: the training-event contract written by the B2.5 fork (what each
  event holds, where the surrogate boundary sits, how events are validated).
- `eirene_return_dataflow.{svg,png}`: what B2.5 receives from EIRENE and what it does with it.
- `solps_eirene_solstice_coupling_note.{tex,pdf}`: the coupling note (method and first results).
- `handoffs/`: dated working notes between sessions and machines. `nersc_eirene_sources.md` is the
  live one for training on Perlmutter (status sections are appended, newest last);
  `2026-10-06_mora_training_dump.md` records the Mora build/run recipes and the flat-start pitfall.

How to run things is in the top-level README and in `jobs/` (one spec per workflow stage).

Notebooks (`notebooks/`): `eirene_dump_explorer` (event files), `fort31_explorer` (legacy background
record), `hero_reference_run` + `hero_tools.py` (the reference SOLPS-ITER run: convergence, wall time,
final-state maps, checkpoint-to-checkpoint change; data expected under
`~/data/eirene_nn/reference_uniform_dt1e-5/`, mesh geometry from the SOLSTICE mesh store).
