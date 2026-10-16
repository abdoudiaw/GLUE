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
wait; echo "$(date): all lanes done; copying states and logs to Dropbox"
cd /pscratch/sd/d/diawa/eirene_nn/coupling && ~/.local/bin/rclone copy --tpslimit 8 --transfers 4 --include "states/**" --include "segments/**" \
  --include "decisions.log" --include "service.jsonl" --include "service.out" --include "retrain_loop.log" --include "comparison*.txt" \
  --exclude "work/**" --exclude "events/**" --exclude "requests/**" --exclude "gated_try1/**" --exclude "surrogate_only_gatedbymistake/**" \
  runs/ dropbox:SOLPS_DB/coupling_results_20261015/runs/ && echo "$(date): Dropbox copy done"
