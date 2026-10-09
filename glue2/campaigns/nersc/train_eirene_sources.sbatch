#!/bin/bash
# Train the SOLSTICE sources model (EIRENE return from the plasma background) on the
# latest GLUE2 snapshot, from the GLUE checkout: `glue2 train` runs the trainer in
# external/solstice. One A100 in the shared queue; resumes from last.pt if resubmitted.
#
# Set-up once:  git clone --recurse-submodules -b glue2-solps-eirene <GLUE> $SCRATCH/GLUE
#               copy ~/data/eirene_nn/{glue2_work,stores} from the Mac to $SCRATCH/eirene_nn
#               edit glue2/configs/diiid_eirene_v3.nersc.example.yaml paths if they differ
#   sbatch glue2/campaigns/nersc/train_eirene_sources.sbatch
#   NAME=eirene_sources_s1 EXTRA="--set seed=1" sbatch glue2/campaigns/nersc/train_eirene_sources.sbatch
#   EPOCHS=2 NAME=smoke bash glue2/campaigns/nersc/train_eirene_sources.sbatch     # smoke test
#SBATCH -A m5186
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -t 4:00:00
#SBATCH -J glue2-eirene-sources
#SBATCH -o logs/eirene/slurm-%x-%j.out
set -euo pipefail
cd ${GLUE_ROOT:-/pscratch/sd/d/diawa/GLUE}
module load ${PYTORCH_MODULE:-pytorch/2.13.0}
source "${SOLSTICE_VENV:-$SCRATCH/venvs/solstice}/bin/activate"
export PYTHONPATH=$PWD/glue2:${PYTHONPATH:-} OMP_NUM_THREADS=8    # the glue2 package is glue2/glue2
mkdir -p logs/eirene

CONFIG=${CONFIG:-glue2/configs/diiid_eirene_v3.nersc.example.yaml}
NAME=${NAME:-eirene_sources_v3}
python -m glue2.cli --config $CONFIG train --name $NAME \
  --set train.epochs=${EPOCHS:-1500} --set train.max_hours=${MAX_HOURS:-3.8} ${EXTRA:-}
