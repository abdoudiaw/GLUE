#!/bin/bash
# Fresh EIRENE labels on drifted states: two CPU interactive lanes over runs/logs/label_list.txt (one short run per state).
cd /pscratch/sd/d/diawa/GLUE; R=/pscratch/sd/d/diawa/eirene_nn/coupling/runs
lane() { while read n; do [ -z "$n" ] && continue; ls $R/$n/states/b2time_* >/dev/null 2>&1 && continue
  NODE_CONSTRAINT=cpu NODE_GPUS="" NSEG=1 bash campaigns/nersc/couple_chunks.sh $n > $R/logs/chunks-$n.out 2>&1 < /dev/null; echo "$(date +%H:%M) $n done"; done; }
awk "NR%2==1" $R/logs/label_list.txt > $R/logs/label_lane_a.txt; awk "NR%2==0" $R/logs/label_list.txt > $R/logs/label_lane_b.txt
lane < $R/logs/label_lane_a.txt & sleep 20; lane < $R/logs/label_lane_b.txt & wait; echo "$(date +%H:%M) all labels done"
