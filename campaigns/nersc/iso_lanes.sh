#!/bin/bash
# Isolation tests of the retained fields: one CPU lane (2 tests in sequence), two GPU 40 GB lanes.
cd /pscratch/sd/d/diawa/GLUE; R=/pscratch/sd/d/diawa/eirene_nn/coupling/runs
cpu() { for n in "$@"; do NODE_CONSTRAINT=cpu NODE_GPUS="" NSEG=1 bash campaigns/nersc/couple_chunks.sh $n > $R/logs/chunks-$n.out 2>&1; echo "$(date +%H:%M) $n done"; done; }
gpu() { for n in "$@"; do NODE_CONSTRAINT="gpu&hbm40g" NSEG=1 bash campaigns/nersc/couple_chunks.sh $n > $R/logs/chunks-$n.out 2>&1; echo "$(date +%H:%M) $n done"; done; }
cpu ${CPU_TESTS} & sleep 10; gpu ${GPU1} & sleep 10; gpu ${GPU2} & wait; echo "$(date +%H:%M) all isolation tests done"
