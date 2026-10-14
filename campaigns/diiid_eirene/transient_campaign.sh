#!/bin/bash
# Transient EIRENE training events along the uniform-start DIII-D trajectory (Mora).
#
# One trajectory is advanced from the uniform state with the training-dump build: at each
# save point the next NB steps are run with the dump on (a training case), the steps up to
# the next save point with the dump off. Around every saved state, SEEDS perturbed copies
# (perturb_b2fstate.py: smooth +-AMP factors on na, ne, te, ti) are each advanced NB steps
# with the dump on. Every case is validated, compressed, given the GLUE2 side-cars, uploaded
# to Dropbox and removed locally (Mora's disk is nearly full); saved states stay in states/.
# Resumable: finished cases and existing states are skipped.
#
# Prerequisite: states/b2fstate_step00050 from the pilot (old 3.0.8 build, 50 steps from the
# uniform state; the training-dump build stops on the flat start itself).
#
#   nohup setsid ./transient_campaign.sh > transient_campaign.log 2>&1 < /dev/null &
#   SAVE="50 100" SEEDS="1" ./transient_campaign.sh          # a short test
set -u
A=/home/cloud/solps-runs/diii-d/acceleration_tests
T=$A/transient_v3
NEW=/home/cloud/local/solps/solps-iter-eirene-training-dump
NP=${NP:-64}; DT=${DT:-1.0e-5}; NB=${NB:-20}
SAVE=${SAVE:-"50 100 200 400 800 1600 3200 6400 12800 20000"}
SEEDS=${SEEDS:-"1 2 3"}; AMP=${AMP:-0.2}; CONST=${CONST:-0.1}
DROPBOX=${DROPBOX:-"dropbox:SOLPS_DB/eirene_training_v3_transient__DIII-D__APP-FPP__uniform_dt1e-5"}
VALIDATOR=$NEW/modules/B2.5/src/test/validate_eirene_training_dump.py
PERTURB=$A/perturb_b2fstate.py
mkdir -p $T/states $T/done $T/chain $T/branch
[ -e $T/baserun ] || ln -s ../baserun $T/baserun
[ -e $T/chain/baserun ] || ln -s ../../baserun $T/chain/baserun
[ -e $T/branch/baserun ] || ln -s ../../baserun $T/branch/baserun

state() { printf "%s/states/b2fstate_step%05d" $T $1; }

stage() {   # stage DIR INIT NTIM DUMP(0|1)
  local d=$1 init=$2 nt=$3 dump=$4
  [ -d $d ] && rm -rf $d
  cp -a $A/template_chi1 $d
  cp -p $init $d/b2fstati
  sed -i -e "s/^'b2mndr_ntim'.*/'b2mndr_ntim'                      '$nt'         # nsteps/" \
         -e "s/^'b2mndr_dtim'.*/'b2mndr_dtim'                      '$DT'      # dt (sec)/" \
         -e "s/^'b2mwti_2dwrite'.*/'b2mwti_2dwrite'               '0'   # new build rejects -1/" $d/b2mn.dat
  grep -q "^'eirene_training_dump'" $d/b2mn.dat || sed -i "/^'b2mndr_ntim'/a'eirene_training_dump'        '$dump'   # save raw EIRENE training events" $d/b2mn.dat
  sed -i "s/^'eirene_training_dump'.*/'eirene_training_dump'        '$dump'/" $d/b2mn.dat
  { echo "init    $init ($(md5sum $init | cut -c1-8))"; echo "dt      $DT"; echo "ntim    $nt"; echo "dump    $dump"; echo "staged  $(date)"; } > $d/RUN_SPEC
  rm -f $d/run.csh
}

run() {     # run DIR -> 0 when the last step completed
  local d=$1
  cat > $d/run_tree.csh <<CSH
#!/bin/tcsh
cd $NEW
setenv SOLPS_HOST_NAME_FORCE ORNL
source setup.csh gfortran
setenv GLI_HOME /home/cloud/local/solps/solps-libs
setenv MSCL_ROOT /home/cloud/local/solps/solps-libs/lib
setenv HWLOC_COMPONENTS "-gl"
if (\$?LD_LIBRARY_PATH) then
  setenv LD_LIBRARY_PATH /home/cloud/local/solps/solps-libs/lib:\${LD_LIBRARY_PATH}
else
  setenv LD_LIBRARY_PATH /home/cloud/local/solps/solps-libs/lib
endif
cd $d
touch b2mn.dat
touch b2fstati
b2run -m \"mpirun --bind-to none -x HWLOC_COMPONENTS -x LD_LIBRARY_PATH -np $NP\" b2mn >& run.log
exit \$status
CSH
  local nt t0 t1
  nt=$(sed -n "s/^'b2mndr_ntim' *'\([0-9]*\)'.*/\1/p" $d/b2mn.dat)
  t0=$(date +%s); tcsh $d/run_tree.csh; t1=$(date +%s)
  local iters; iters=$(grep -c '^ *ITER' $d/run.log)
  echo "   $(basename $d): $((t1-t0)) s for $nt steps, ITER lines $iters, b2fstate $(stat -c %s $d/b2fstate 2>/dev/null || echo 0) bytes"
  grep -iE "supra|xerrab|Exit code" $d/run.log | head -2
  [ -s $d/b2fstate ] && [ "$(stat -c %s $d/b2fstate)" -gt 1000000 ] && grep -q "^ *ITER *$nt " $d/run.log
}

package() {  # package DIR STEP SEED PERTURB_JSON -> uploaded and removed
  local d=$1 s=$2 seed=$3 pj=$4 case; case=$(basename $d)
  cd $d
  python3 $VALIDATOR . > eirene_training_validation.log 2>&1 || { echo "   $case: VALIDATION FAILED"; return 1; }
  for f in eirene_training_v3_*.nc; do nccopy -d4 -s $f $f.tmp && mv $f.tmp $f || return 1; done
  sha256sum eirene_training_v3_*.nc > eirene_training_v3.sha256
  python3 - "$s" "$seed" "$pj" <<'PY'
import json, sys
s, seed, pj = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
meta = json.load(open(pj)) if pj != "-" else {}
json.dump({"inputs": {"transient": {"step": s, "seed": seed, "amplitude": meta.get("amplitude", 0.0),
                                    "constant": meta.get("constant", 0.0)}},
           "trajectory": "uniform_dt1e-5 (DIII-D APP-FPP, chi 1, D 0.3, 3+3 MW, 7.5e20, puff 2.5e21)",
           "perturbation": meta}, open("source_params.json", "w"), indent=1)
PY
  { echo "restart_source=b2fstati"; git -C $NEW rev-parse HEAD; git -C $NEW/modules/B2.5 rev-parse HEAD; } > eirene_training_provenance.txt
  touch EIRENE_TRAINING_SUCCESS
  rclone copy --include "eirene_training_v3*" --include "source_params.json" --include "eirene_training_provenance.txt" \
     --include "EIRENE_TRAINING_SUCCESS" --include "b2mn.dat" --include "RUN_SPEC" --include "run.log" . $DROPBOX/$case \
     && rclone check --include "eirene_training_v3_*.nc" . $DROPBOX/$case > /dev/null 2>&1 \
     || { echo "   $case: UPLOAD FAILED (kept locally)"; return 1; }
  cd $T && rm -rf $d && touch $T/done/$case
  echo "   $case: uploaded $(rclone lsf $DROPBOX/$case | grep -c '\.nc$') events"
}

echo "=== transient campaign start $(date)  save points: $SAVE ; seeds: $SEEDS ; NB $NB ; NP $NP"
[ -s "$(state 50)" ] || { echo "missing $(state 50) (run transient_pilot.sh first)"; exit 2; }
prev=50
for s in $SAVE; do
  # dump segment at the save point prev (the trajectory continues through it)
  case=$(printf "run_tr%05d_s0__D" $prev)
  if [ ! -e $T/done/$case ]; then
    echo "--- chain: dump segment at step $prev  $(date)"
    stage $T/chain/$case "$(state $prev)" $NB 1
    run $T/chain/$case && cp -p $T/chain/$case/b2fstate "$(state $((prev+NB)))" && package $T/chain/$case $prev 0 - \
      || { echo "chain stopped at $case"; exit 3; }
  fi
  # branches: perturbed copies of the saved state
  for seed in $SEEDS; do
    case=$(printf "run_tr%05d_s%d__D" $prev $seed)
    [ -e $T/done/$case ] && continue
    echo "--- branch: step $prev seed $seed  $(date)"
    pst=$(state $prev)_s$seed
    python3 $PERTURB "$(state $prev)" $pst --seed $((prev*100+seed)) --amplitude $AMP --constant $CONST > /dev/null || { echo "perturbation failed"; exit 3; }
    stage $T/branch/$case $pst $NB 1
    run $T/branch/$case && package $T/branch/$case $prev $seed $pst.perturbation.json || echo "branch $case failed (continuing)"
  done
  # gap to the next save point, dump off
  [ $s -le $prev ] && continue
  if [ ! -s "$(state $s)" ]; then
    gap=$((s - prev - NB))
    echo "--- chain: $gap steps from $((prev+NB)) to $s, dump off  $(date)"
    stage $T/chain/gap_$s "$(state $((prev+NB)))" $gap 0
    run $T/chain/gap_$s && cp -p $T/chain/gap_$s/b2fstate "$(state $s)" && rm -rf $T/chain/gap_$s || { echo "chain stopped in gap to $s"; exit 3; }
  fi
  prev=$s
done
# the last save point gets its dump segment and branches too
case=$(printf "run_tr%05d_s0__D" $prev)
if [ ! -e $T/done/$case ]; then
  stage $T/chain/$case "$(state $prev)" $NB 1
  run $T/chain/$case && package $T/chain/$case $prev 0 - || echo "final dump segment failed"
fi
for seed in $SEEDS; do
  case=$(printf "run_tr%05d_s%d__D" $prev $seed); [ -e $T/done/$case ] && continue
  pst=$(state $prev)_s$seed
  python3 $PERTURB "$(state $prev)" $pst --seed $((prev*100+seed)) --amplitude $AMP --constant $CONST > /dev/null
  stage $T/branch/$case $pst $NB 1
  run $T/branch/$case && package $T/branch/$case $prev $seed $pst.perturbation.json || echo "branch $case failed"
done
df -h / | tail -1
echo "=== transient campaign done $(date): $(ls $T/done | wc -l) cases"
