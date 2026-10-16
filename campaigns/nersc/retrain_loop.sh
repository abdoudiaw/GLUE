#!/bin/bash
# Retrain loop beside the gated coupled run, on the same Perlmutter node (GPUs 1-3), one GLUE2 cycle
# every INTERVAL seconds: ingest the run's fallback events, fine-tune when retrain_min_new new events
# exist, calibrate, promote, write the reload file the service watches. Stop with a file named STOP
# in the run directory.
#   nohup bash campaigns/nersc/retrain_loop.sh configs/nersc_coupling.yaml > runs/logs/retrain_loop.log 2>&1 &
set -u
CONFIG=${1:?config}; INTERVAL=${INTERVAL:-600}
RUN_DIR=$(python - "$CONFIG" <<'PY'
import sys, yaml, os; print(os.path.dirname(yaml.safe_load(open(sys.argv[1]))["reload_file"]))
PY
)
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
export PYTHONPATH=$PWD:$PWD/external/solstice/src CUDA_VISIBLE_DEVICES=${RETRAIN_GPUS:-1,2,3}
while [ ! -e "$RUN_DIR/STOP" ]; do
  echo "=== cycle $(date)"
  python -m glue2.cli --config "$CONFIG" cycle > "$RUN_DIR/retrain_cycle_$(date +%Y%m%d_%H%M%S).log" 2>&1; rc=$?
  tail -20 "$(ls -t "$RUN_DIR"/retrain_cycle_*.log | head -1)"; [ $rc -ne 0 ] && echo "cycle exit $rc: full output in $(ls -t "$RUN_DIR"/retrain_cycle_*.log | head -1)"
  sleep $INTERVAL
done
echo "stopped $(date)"
