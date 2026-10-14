#!/bin/bash
# Transient training-data campaign on Mora (campaigns/diiid_eirene/transient_campaign.sh with
# its parameters from the spec). Resumable; uploads each case to Dropbox and removes it locally.
export SAVE="${save_points}" SEEDS="${seeds}" NB=${dump_steps} NP=${mpi_ranks} AMP=${amplitude} CONST=${constant} DROPBOX="${dropbox}"
cd ${runs_root}
cp -p ${glue_root}/campaigns/diiid_eirene/transient_campaign.sh ${glue_root}/campaigns/diiid_eirene/perturb_b2fstate.py .
./transient_campaign.sh
echo "JOB DONE $(date)"
