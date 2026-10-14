#!/bin/bash
# One coupled SOLPS-ITER run on Mora, staged from the template case:
#   replay 0  plain B2.5-EIRENE (reference-style run)
#   replay 3  surrogate mode, with glue2.surrogate_service answering (policy gate|always|never)
# Outputs under ${case_dir}: b2fstate, run.log, eirene_surrogate_decisions.log, service.jsonl,
# events/ (dumped EIRENE calls), provenance.txt (per call: request, answer, wait, score, novelty).
set -u
D=${case_dir}
[ -d $D ] && [ "${overwrite}" = "1" ] && rm -rf $D
[ -d $D ] && { echo "$D exists (overwrite: 0)"; exit 1; }
cp -a ${template_case} $D; cp -p ${init_state} $D/b2fstati; rm -f $D/run.csh
[ -e $(dirname $D)/baserun ] || ln -s ${runs_root}/baserun $(dirname $D)/baserun
sed -i -e "s/^'b2mndr_ntim'.*/'b2mndr_ntim'                      '${nsteps}'/" \
       -e "s/^'b2mndr_dtim'.*/'b2mndr_dtim'                      '${dt}'/" \
       -e "s/^'b2mwti_2dwrite'.*/'b2mwti_2dwrite'               '0'/" $D/b2mn.dat
sed -i "/^'b2mndr_ntim'/a'eirene_training_dump'        '${dump}'\n'eirene_training_replay'      '${replay}'\n'eirene_surrogate_timeout'    '${timeout}'" $D/b2mn.dat
${extra_b2mn}
echo "init ${init_state} ($(md5sum ${init_state} | cut -c1-8)); steps ${nsteps}; dt ${dt}; replay ${replay}; dump ${dump}; bundle ${bundle}; policy ${policy}; ranks ${mpi_ranks}; staged $(date)" > $D/RUN_SPEC
mkdir -p $D/b2mn.exe.dir
cat > $D/run_tree.csh <<'CSH'
#!/bin/tcsh
cd ${solps_tree}
${launch_env_tcsh}
cd ${case_dir}
touch b2mn.dat; touch b2fstati
b2run -m \"mpirun --bind-to none -x HWLOC_COMPONENTS -x LD_LIBRARY_PATH -np ${mpi_ranks}\" b2mn >& run.log
CSH
svc=""
if [ "${replay}" = "3" ] && [ "${policy}" != "none" ]; then
  export PYTHONPATH=${glue_root}:${glue_root}/external/solstice/src SOLSTICE_DATA=${data_root}/stores OMP_NUM_THREADS=${omp_threads}
  pol=""; [ "${policy}" = "always" ] && pol=--always; [ "${policy}" = "never" ] && pol=--never
  ${python} -m glue2.surrogate_service --bundle ${bundle} --run-dir $D/b2mn.exe.dir --device cpu $pol --log $D/service.jsonl --reload-file $D/reload_bundle --keep-requests $D/requests ${service_args} > $D/service.out 2>&1 &
  svc=$!; sleep ${service_warmup}
fi
echo "RUN START $(date)"
t0=$(date +%s); tcsh $D/run_tree.csh; t1=$(date +%s)
[ -n "$svc" ] && kill $svc 2>/dev/null
echo "RUN EXIT $((t1-t0)) s for ${nsteps} steps, ITER lines $(grep -c '^ *ITER' $D/run.log), b2fstate $(stat -c %s $D/b2fstate 2>/dev/null || echo 0) bytes"
grep -iE "supra|xerrab|Exit code|no answer" $D/run.log | head -3
mv $D/b2mn.exe.dir/eirene_surrogate_decisions.log $D/ 2>/dev/null
mkdir -p $D/events; mv $D/eirene_training_v3_*.nc $D/b2mn.exe.dir/eirene_training_v3_*.nc $D/events/ 2>/dev/null
if [ -f $D/eirene_surrogate_decisions.log ]; then
python3 - $D <<'PY'
import json, sys, glob, os
d = sys.argv[1]; dec = {}
for line in open(f"{d}/eirene_surrogate_decisions.log"):
    c, w, u, t = line.split(); dec[int(c)] = (w == "T", u == "T", float(t))
svc = {}
if os.path.exists(f"{d}/service.jsonl"):
    for line in open(f"{d}/service.jsonl"):
        r = json.loads(line); svc[r.get("call")] = r
events = sorted(glob.glob(f"{d}/events/*.nc"))
n_sur = sum(1 for v in dec.values() if v[1])
out = [f"calls {len(dec)}: surrogate {n_sur}, EIRENE {len(dec) - n_sur}; events dumped {len(events)}",
       f"{'call':>4s} {'request':>7s} {'answer':>7s} {'wait_s':>7s} {'score':>7s} {'novelty':>8s} {'gate_ok':>7s}  source"]
for c in sorted(dec):
    w, u, t = dec[c]; r = svc.get(c, {})
    out.append(f"{c:4d} {str(w):>7s} {str(u):>7s} {t:7.2f} {r.get('score', float('nan')):7.2f} {r.get('novelty', float('nan')):8.2f} {str(r.get('ok', '')):>7s}  {'surrogate' if u else 'EIRENE'}")
open(f"{d}/provenance.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
PY
fi
df -h / | tail -1
echo "JOB DONE $(date)"
