#!/bin/bash
# Overnight training-data lanes (interactive QOS, 2 slots). Each lane waits for the one-source test occupying its
# slot to finish (3 states or its allocation gone), then runs plain EIRENE with event dumping.
set -u
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
R=/pscratch/sd/d/diawa/eirene_nn/coupling/runs
waitfor() { until [ "$(ls $R/$1/states 2>/dev/null | grep -c state_)" -ge "$2" ] || ! squeue -h -n $1 -u diawa | grep -q .; do sleep 60; done; echo "$(date): $1 finished"; }
( waitfor anchor2_only_sni 3; for r in eirene_uniform_dump:8 eirene_uniform_cont:7; do echo "$(date): lane A starts ${r%%:*}"; NSEG=${r##*:} bash campaigns/nersc/couple_chunks.sh ${r%%:*} > $R/logs/chunks-${r%%:*}.out 2>&1; echo "$(date): lane A finished ${r%%:*}"; done ) &
( waitfor anchor2_only_energy 3; echo "$(date): lane B starts eirene_nn_v2_dump"; NSEG=10 bash campaigns/nersc/couple_chunks.sh eirene_nn_v2_dump > $R/logs/chunks-eirene_nn_v2_dump.out 2>&1; echo "$(date): lane B finished eirene_nn_v2_dump" ) &
wait; echo "$(date): data lanes done"
