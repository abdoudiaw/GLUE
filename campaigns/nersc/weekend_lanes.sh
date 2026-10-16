#!/bin/bash
# Weekend matrix on the interactive QOS (2 lanes). Run on a Perlmutter LOGIN node from $SCRATCH/GLUE:
#   nohup campaigns/nersc/weekend_lanes.sh > $SCRATCH/eirene_nn/coupling/runs/logs/weekend_lanes.out 2>&1 &
# Edit the two lists below to choose the runs (name:nseg). All start from run 3's final bundle 41c6d56fe16d.
set -u
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
LOGS=/pscratch/sd/d/diawa/eirene_nn/coupling/runs/logs
LANE_A="gated_anchor20:15 gated_anchor20_noretrain:15 gated_f10:15"
LANE_B="gated_nn_v2_anchor20:10 surrogate_only_anchor20:15 gated_anchor50:15"
lane() { tag=$1; shift; for r in "$@"; do echo "$(date): lane $tag starts ${r%%:*}"; NSEG=${r##*:} bash campaigns/nersc/couple_chunks.sh ${r%%:*} > $LOGS/chunks-${r%%:*}.out 2>&1; echo "$(date): lane $tag finished ${r%%:*}"; done; }
lane A $LANE_A & lane B $LANE_B &
wait; echo "$(date): all lanes done; copying states and logs to Dropbox"
cd /pscratch/sd/d/diawa/eirene_nn/coupling && ~/.local/bin/rclone copy --tpslimit 8 --transfers 4 --include "states/**" --include "segments/**" \
  --include "decisions.log" --include "service.jsonl" --include "service.out" --include "retrain_loop.log" --include "*.txt" \
  --exclude "work/**" --exclude "events/**" --exclude "requests/**" --exclude "gated_try1/**" --exclude "surrogate_only_gatedbymistake/**" \
  runs/ dropbox:SOLPS_DB/coupling_results_20261015/runs/ && echo "$(date): Dropbox copy done"
