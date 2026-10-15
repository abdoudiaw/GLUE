#!/bin/bash
# Render a spec whose bundle is "[BUNDLE]" with the catalog's currently promoted bundle, then start it
# through the interactive chunk loop.   usage: launch_with_promoted.sh <spec basename without .yaml> <nseg>
set -u
SPEC=${1:?spec}; NSEG=${2:-15}
cd ${GLUE_ROOT:-$SCRATCH/GLUE}
module load pytorch/2.13.0 > /dev/null 2>&1; source $SCRATCH/venvs/solstice/bin/activate
export PYTHONPATH=$PWD:$PWD/external/solstice/src
P=$(python -c 'from glue2.catalog import Catalog
with Catalog.reader("/pscratch/sd/d/diawa/eirene_nn/glue2_work/diiid_eirene_v3/catalog.sqlite") as cat:
    b = cat.promoted_bundle(); print(b["path"] if b is not None else "")')
[ -n "$P" ] || { echo "no promoted bundle in the catalog"; exit 1; }
NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open('jobs/specs/$SPEC.yaml'))['name'])")
sed "s#\"\[BUNDLE\]\"#$P#" jobs/specs/$SPEC.yaml > jobs/specs/.$SPEC.filled.yaml
CFG=$(python -c "import yaml; print((yaml.safe_load(open('jobs/specs/$SPEC.yaml')).get('params') or {}).get('retrain_config',''))")
[ -n "$CFG" ] && [ -f "$CFG" ] && sed -i "s#\"\[BUNDLE\]\"#$P#" "$CFG" && echo "retrain config $CFG warm-starts from $P"
python -m glue2.jobs render jobs/specs/.$SPEC.filled.yaml > /dev/null || exit 1
echo "$NAME: bundle $P; rendered jobs/rendered/$NAME.sh"
NSEG=$NSEG nohup campaigns/nersc/couple_chunks.sh $NAME > /pscratch/sd/d/diawa/eirene_nn/coupling/runs/logs/chunks-$NAME.out 2>&1 &
echo "chunk loop started (pid $!)"
