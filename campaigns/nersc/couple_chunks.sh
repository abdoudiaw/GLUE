#!/bin/bash
# Run a coupled job (jobs/rendered/<name>.sh) to completion through the interactive QOS, one 4-h chunk
# after another (campaigns/nersc/couple_salloc.sh), until all ${NSEG:-15} states exist or a file
# named STOP_CHUNKS appears in the run directory. Survives the user logging out (nohup), not the
# death of the node it runs on; relaunch it then.
#   nohup campaigns/nersc/couple_chunks.sh <name> > coupling/runs/logs/chunks-<name>.out 2>&1 &
set -u
NAME=${1:?rendered job name}; NSEG=${NSEG:-15}; WALL=${WALL:-04:00:00}
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
OUT=/pscratch/sd/d/diawa/eirene_nn/coupling/runs/$NAME
n=1
while true; do
  have=$(ls $OUT/states 2>/dev/null | grep -c '^state_')
  [ "$have" -ge "$NSEG" ] && { echo "$(date): $have states, done"; break; }
  [ -e $OUT/STOP_CHUNKS ] && { echo "$(date): STOP_CHUNKS present, stopping"; break; }
  if [ -s $OUT/RUNNING ] && squeue -h -j "$(cat $OUT/RUNNING)" 2>/dev/null | grep -q .; then
    echo "$(date): run owned by job $(cat $OUT/RUNNING) (a batch copy); waiting 10 min"; sleep 600; continue
  fi
  echo "$(date): chunk $n ($have of $NSEG states so far)"
  bash campaigns/nersc/couple_salloc.sh $NAME $WALL > /pscratch/sd/d/diawa/eirene_nn/coupling/runs/logs/salloc-$NAME-auto$n.out 2>&1
  echo "$(date): chunk $n ended (exit $?)"
  n=$((n+1)); sleep 60
done
