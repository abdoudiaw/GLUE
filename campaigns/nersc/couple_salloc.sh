#!/bin/bash
# Run a rendered coupled job (jobs/rendered/<name>.sh) on an interactive Perlmutter GPU node instead of
# the regular queue: salloc (interactive QOS: high priority, starts in seconds, 4 h max, 2 per user)
# run the script where salloc runs (a login or Jupyter node): the service and B2.5+EIRENE are sibling
# job steps on the allocated node (specs set service_prefix and exchange_dir on scratch for this), the
# retrain loop runs beside the driver. Running the script itself as a step did not work reliably
# (steps that never start, nested-step CPU limits). 80 GB A100 nodes only (hbm80g) for uniform timing.
# The script skips segments whose state already exists, so rerun it for the next 4 h.
#   nohup campaigns/nersc/couple_salloc.sh <name> [walltime=04:00:00] > coupling/runs/logs/salloc-<name>.out 2>&1 &
set -u
NAME=${1:?rendered job name}; WALL=${2:-04:00:00}
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
# drop the SLURM variables of the job this shell runs in (a Jupyter node): steps inside the new
# allocation inherit e.g. SLURM_MEM_PER_CPU and then fail on nodes with less memory
unset $(env | grep -oE "^SLURM_[A-Za-z0-9_]+" | tr "\n" " ") 2>/dev/null
# NODE_CONSTRAINT=cpu NODE_GPUS="" runs the case on a CPU interactive node (separate quota from the GPU
# interactive jobs; the service then needs service_device: cpu in the spec, ~0.2 s per call with 16 threads)
exec salloc -A m5186 -q interactive -C "${NODE_CONSTRAINT:-gpu&hbm80g}" -N 1 --ntasks-per-node=64 --cpus-per-task=2 ${NODE_GPUS---gpus-per-node=4} -t $WALL -J $NAME \
  bash jobs/rendered/$NAME.sh
