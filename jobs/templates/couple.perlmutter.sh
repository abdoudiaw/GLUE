#!/bin/bash
# Coupled SOLPS-ITER run with the EIRENE surrogate on one Perlmutter GPU node, as chained 1000-step
# segments from START_STATE, keeping the plasma state at every segment end (same layout as the Mora
# hero run). Resume: submit again with START_STATE = the last saved state.
#
#   sbatch ${name}.sh                                   # from ${start_state}
#   sbatch --export=ALL,START_STATE=<path> ${name}.sh   # resume
#
# One node: B2.5 (serial) + EIRENE (${mpi_ranks} ranks) on the CPU cores; the GLUE service on GPU 0;
# the retrain loop (a separate script, see docs/handoffs/nersc_coupled_runs.md) on GPUs 1-3.
# The case runs in node-local storage (${exchange_dir}); states, events, logs are copied to Lustre
# at every segment end.
#SBATCH -A ${account}
#SBATCH -C ${constraint}
#SBATCH -q ${qos}
#SBATCH -N 1
#SBATCH --ntasks-per-node=${mpi_ranks}
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH -t ${walltime}
#SBATCH -J ${name}
#SBATCH -o ${coupling_root}/runs/logs/slurm-%x-%j.out
set -u
NAME=${name}
OUT=${coupling_root}/runs/$NAME            # Lustre: states/, events/, segments/ (logs), provenance
WORK=${exchange_dir}/$NAME                 # node-local: the running case
START=${START_STATE:-${start_state}}
NSEG=${nseg}; SEG=${seg_steps}; DT=${dt}
mkdir -p $OUT/states $OUT/events $OUT/segments $OUT/requests ${coupling_root}/runs/logs $WORK
# one driver per run: a batch job and an interactive chunk must never work on the same run directory
if [ -s $OUT/RUNNING ] && [ "$(cat $OUT/RUNNING)" != "${SLURM_JOB_ID:-}" ] && squeue -h -j "$(cat $OUT/RUNNING)" 2>/dev/null | grep -q .; then
  echo "$NAME is already being run by job $(cat $OUT/RUNNING); exiting"; exit 0
fi
echo "${SLURM_JOB_ID:-}" > $OUT/RUNNING
cd ${solps_tree}
set +u    # the SOLPS setup references unset variables
${launch_env_bash}
set -u
cd $WORK
module load ${pytorch_module}
source ${venv}/bin/activate
export PYTHONPATH=${glue_root}:${glue_root}/external/solstice/src SOLSTICE_DATA=${data_root}/stores OMP_NUM_THREADS=${omp_threads}

# the bundle: the spec's, or (retrain loop) the bundle the catalog has promoted since, so that a later
# chunk of a run resumed with the same script continues with the retrained model
BUNDLE=${bundle}
if [ -n "${retrain_config}" ]; then
  P=$(python -c 'import sys, yaml
from glue2.catalog import Catalog
cfg = yaml.safe_load(open(sys.argv[1]))
with Catalog.reader(cfg["workdir"] + "/catalog.sqlite") as cat:
    b = cat.promoted_bundle()
    print(b["path"] if b is not None else "")' "${retrain_config}")
  [ -n "$P" ] && BUNDLE=$P
fi
echo "bundle: $BUNDLE"
# the service, on GPU 0, watching the case's b2mn.exe.dir for the whole job
mkdir -p $WORK/case/b2mn.exe.dir
CUDA_VISIBLE_DEVICES=0 ${service_prefix} python -m glue2.surrogate_service --bundle $BUNDLE --run-dir $WORK/case/b2mn.exe.dir \
   --device cuda --log $OUT/service.jsonl --reload-file $OUT/reload_bundle \
   --keep-requests $OUT/requests --keep-every ${keep_every} ${policy_flag} ${service_args} > $OUT/service.out 2>&1 &
SVC=$!
sleep ${service_warmup}
# ship dumped EIRENE events to $OUT/events while a segment runs (files untouched for 2 min are closed),
# so the retrain loop can act within a segment instead of only at its end
( while true; do sleep 120; d=$OUT/events/$(cat $WORK/SEGTAG 2>/dev/null || echo seg_unknown); mkdir -p $d; find $WORK/case/b2mn.exe.dir $WORK/case -maxdepth 1 -name 'eirene_training_v3_*.nc' -mmin +2 -exec mv -t $d/ {} + 2>/dev/null; done ) &
SHIP=$!
# the retrain loop (gated run): a separate process on GPUs 1-3, one glue2 cycle every retrain_interval s
RL=""
if [ -n "${retrain_config}" ]; then
  rm -f $OUT/STOP    # left by a previous chunk of this run
  INTERVAL=${retrain_interval} RETRAIN_GPUS=${retrain_gpus} GLUE_ROOT=${glue_root} \
    nohup ${retrain_prefix} bash ${glue_root}/campaigns/nersc/retrain_loop.sh ${retrain_config} > $OUT/retrain_loop.log 2>&1 &
  RL=$!
fi

step=$(basename $START | sed 's/.*step0*//; s/.*state_0*//; s/[^0-9].*//')   # ..._stepNNNNN or state_NNNNN names carry the step
[ -z "$step" ] && step=50
prev_state=$START
echo "=== $NAME start $(date): from $START (step $step), $NSEG x $SEG steps"
for i in $(seq 1 $NSEG); do
  s=$((step + SEG))
  tag=$(printf %05d $s)
  [ -s $OUT/states/state_$tag ] && { step=$s; prev_state=$OUT/states/state_$tag; continue; }
  rm -rf $WORK/case; cp -a ${coupling_root}/template_chi1 $WORK/case; cp -p $prev_state $WORK/case/b2fstati; echo seg_$tag > $WORK/SEGTAG
  ln -sfn ${coupling_root}/baserun $WORK/baserun
  cd $WORK/case
  sed -i -e "s/^'b2mndr_ntim'.*/'b2mndr_ntim'                      '$SEG'/" \
         -e "s/^'b2mndr_dtim'.*/'b2mndr_dtim'                      '$DT'/" \
         -e "s/^'b2mwti_2dwrite'.*/'b2mwti_2dwrite'               '0'/" b2mn.dat
  sed -i "/^'b2mndr_ntim'/a'eirene_training_dump'        '${dump}'\n'eirene_training_replay'      '${replay}'\n'eirene_surrogate_timeout'    '${timeout}'" b2mn.dat
  mkdir -p b2mn.exe.dir
  touch b2mn.dat; touch b2fstati
  t0=$(date +%s)
  ${b2run_line}
  t1=$(date +%s)
  it=$(grep -c '^ *ITER' run.log)
  echo "segment to step $s: $((t1-t0)) s, ITER lines $it, b2fstate $(stat -c %s b2fstate 2>/dev/null || echo 0) bytes  $(date)"
  grep -iE "supra|xerrab|no answer" run.log | head -2
  if [ -s b2fstate ] && [ "$(stat -c %s b2fstate)" -gt 1000000 ]; then
    cp -p b2fstate $OUT/states/state_$tag; cp -p b2time.nc $OUT/states/b2time_$tag.nc 2>/dev/null; cp -p b2tallies.nc $OUT/states/b2tallies_$tag.nc 2>/dev/null
    cp -p run.log $OUT/segments/run_$tag.log
    mkdir -p $OUT/events/seg_$tag; mv eirene_training_v3_*.nc b2mn.exe.dir/eirene_training_v3_*.nc $OUT/events/seg_$tag/ 2>/dev/null
    [ -f b2mn.exe.dir/eirene_surrogate_decisions.log ] && cat b2mn.exe.dir/eirene_surrogate_decisions.log >> $OUT/decisions.log
  else
    cp -p run.log $OUT/segments/run_$tag.FAILED.log 2>/dev/null; cp -p b2mn.exe.dir/eirene_surrogate_decisions.log $OUT/segments/decisions_$tag.FAILED.log 2>/dev/null
    echo "segment to step $s FAILED; stopping"; kill $SVC $SHIP; [ -n "$RL" ] && { touch $OUT/STOP; kill $RL; }; exit 3
  fi
  step=$s; prev_state=$OUT/states/state_$tag
done
kill $SVC $SHIP 2>/dev/null
[ -n "$RL" ] && { touch $OUT/STOP; kill $RL 2>/dev/null; }
echo "=== $NAME done $(date): states $(ls $OUT/states | grep -c state_)"
