#!/bin/bash
# Run the remaining coupled cases unattended through the interactive QOS (max 2 at a time): two lanes,
# each a sequence of chunk loops. Run it on a Perlmutter LOGIN node (persists), from $SCRATCH/GLUE:
#   nohup campaigns/nersc/overnight_lanes.sh > $SCRATCH/eirene_nn/coupling/runs/logs/overnight_lanes.out 2>&1 &
# Lane A: run 4b (gated from the NN v2 state), then 4a nearest-case, then 4a v1.
# Lane B: finish 4a v2, then run 2b (surrogate everywhere with run 3's bundle).
set -u
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
LOGS=/pscratch/sd/d/diawa/eirene_nn/coupling/runs/logs
( for r in gated_nn_v2:10 nn_init_nearest_eirene:10 nn_init_v1_eirene:10; do
    echo "$(date): lane A starts ${r%%:*}"; NSEG=${r##*:} bash campaigns/nersc/couple_chunks.sh ${r%%:*} > $LOGS/chunks-${r%%:*}.out 2>&1
    echo "$(date): lane A finished ${r%%:*}"; done ) &
( for r in nn_init_v2_eirene:10 surrogate_only_v2:15; do
    echo "$(date): lane B starts ${r%%:*}"; NSEG=${r##*:} bash campaigns/nersc/couple_chunks.sh ${r%%:*} > $LOGS/chunks-${r%%:*}.out 2>&1
    echo "$(date): lane B finished ${r%%:*}"; done ) &
wait; echo "$(date): all lanes done"
