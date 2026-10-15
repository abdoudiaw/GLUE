#!/bin/bash
# Run a rendered coupled job (jobs/rendered/<name>.sh) on an interactive Perlmutter GPU node instead of
# the regular queue: salloc (interactive QOS: high priority, starts in seconds, 4 h max, 2 per user)
# and run the script on the node as one step; the script's own srun (B2.5+EIRENE) nests inside it
# with --overlap. The script skips segments whose state already exists, so rerun it for the next 4 h.
#   nohup campaigns/nersc/couple_salloc.sh <name> [walltime=04:00:00] > coupling/runs/logs/salloc-<name>.out 2>&1 &
set -u
NAME=${1:?rendered job name}; WALL=${2:-04:00:00}
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
exec salloc -A m5186 -q interactive -C gpu -N 1 --ntasks-per-node=64 --cpus-per-task=2 --gpus-per-node=4 -t $WALL -J $NAME \
  srun -N 1 -n 1 -c 2 --overlap --gpus-per-node=4 bash jobs/rendered/$NAME.sh
