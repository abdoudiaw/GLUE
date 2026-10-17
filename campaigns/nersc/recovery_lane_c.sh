#!/bin/bash
# Third recovery lane: the first uniform segment (steps 50-1050). The interactive QOS allows two jobs per
# user, so this lane waits until recovery_lanes.sh has finished before taking a slot.
cd /pscratch/sd/d/diawa/GLUE; R=/pscratch/sd/d/diawa/eirene_nn/coupling/runs
until grep -q "recovery lanes done" $R/logs/recovery_lanes.out 2>/dev/null; do sleep 120; done
echo "$(date): lane C starts dump_u_00050"
NSEG=1 bash campaigns/nersc/couple_chunks.sh dump_u_00050 > $R/logs/chunks-dump_u_00050.out 2>&1
echo "$(date): lane C finished dump_u_00050"
