#!/bin/bash
# Regenerate the overwritten dumped events: one 1000-step EIRENE segment per run, two lanes on the interactive QOS.
set -u
cd ${GLUE_ROOT:-$SCRATCH/GLUE}; R=/pscratch/sd/d/diawa/eirene_nn/coupling/runs
lane() { tag=$1; shift; for n in "$@"; do rm -f $R/$n/STOP_CHUNKS 2>/dev/null; echo "$(date): lane $tag starts $n"; NSEG=1 bash campaigns/nersc/couple_chunks.sh $n > $R/logs/chunks-$n.out 2>&1; echo "$(date): lane $tag finished $n"; done; }
lane A dump_u_01050 dump_u_03050 dump_u_05050 dump_n_00050 dump_n_04050 &
lane B dump_u_02050 dump_u_04050 dump_u_06050 dump_n_02050 dump_n_06050 &
wait; echo "$(date): recovery lanes done"
