# Campaigns: producing training events on the SOLPS side

`diiid_eirene/` (Mora, DIII-D APP-FPP pure-deuterium case):

- `build_eirene_training_dump.csh`: build the training-dump fork (also available as the `build`
  job stage).
- `run_eirene_training_case.csh`, `generate_eirene_training_db.sh`,
  `generate_eirene_training_db_cloud.sh`: one-step restarts of the archived converged ensemble
  (Dropbox in, events out); this produced the 761-case schema-3 set.
- `transient_campaign.sh` + `perturb_b2fstate.py`: the uniform-start trajectory with dump
  segments at save points and perturbed restarts around each saved state (the `campaign` job stage).
- `validate_eirene_training_dump.py`: schema/shape/finiteness checks of event files.
- `compare_b2fstate.py`: field-by-field comparison of two plasma states (replay and surrogate tests).

`nersc/`: `train_eirene_sources.sbatch`, the Perlmutter job behind `glue2 train`.

Every case directory a campaign produces carries `eirene_training_v3.sha256`,
`source_params.json` and `EIRENE_TRAINING_SUCCESS`, which is what `glue2 ingest` requires.
