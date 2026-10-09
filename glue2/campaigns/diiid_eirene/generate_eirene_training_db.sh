#!/usr/bin/env bash
# Project: SOLPS-ITER/EIRENE training-data generation for SOLSTICE
# Author: Abdou Diaw, Oak Ridge National Laboratory
#
# Mac-side controller. It stages archived cases on Mora, runs the one-step
# EIRENE dump workflow, copies only training products back to Dropbox,
# verifies checksums locally, and removes verified temporary remote cases.

set -euo pipefail

source_root="/Users/42d/Library/CloudStorage/Dropbox-ORNL/Abdou Diaw/SOLPS_DB/ens__DIII-D__APP-FPP__D_C_Ne__ss__lhs__20250124_144649"
output_root="/Users/42d/Library/CloudStorage/Dropbox-ORNL/Abdou Diaw/SOLPS_DB/eirene_training_v3__ens__DIII-D__APP-FPP__D_C_Ne__ss__lhs__20250124_144649"
remote_host="mora"
remote_root="/home/cloud/solps-runs/diii-d/eirene_training_campaign_v3"
remote_base="/home/cloud/solps-runs/diii-d/baserun"
mpi_ranks=64
keep_remote=0
run_all=0
limit=0
cases=()

usage() {
    cat <<'EOF'
Usage:
  generate_eirene_training_db.sh --case RUN_NAME [options]
  generate_eirene_training_db.sh --all [--limit N] [options]

Options:
  --case RUN_NAME   Process one run_* directory; may be repeated.
  --all             Process every top-level run_* directory.
  --limit N         Process at most N selected cases (useful for a pilot).
  --np N            MPI ranks per case (default: 64).
  --keep-remote     Keep successfully transferred Mora working directories.
  --source-root P   Override the archived ensemble directory.
  --output-root P   Override the local training database directory.
  -h, --help        Show this help.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --case)
            cases+=("$2")
            shift 2
            ;;
        --all)
            run_all=1
            shift
            ;;
        --limit)
            limit="$2"
            shift 2
            ;;
        --np)
            mpi_ranks="$2"
            shift 2
            ;;
        --keep-remote)
            keep_remote=1
            shift
            ;;
        --source-root)
            source_root="$2"
            shift 2
            ;;
        --output-root)
            output_root="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if (( run_all == 1 && ${#cases[@]} > 0 )); then
    echo "ERROR: use either --all or explicit --case arguments" >&2
    exit 2
fi

if (( run_all == 1 )); then
    while IFS= read -r case_path; do
        cases+=("$(basename "$case_path")")
    done < <(find "$source_root" -mindepth 1 -maxdepth 1 -type d -name 'run_*' -print | sort)
fi

if (( ${#cases[@]} == 0 )); then
    echo "ERROR: no cases selected" >&2
    usage >&2
    exit 2
fi

if ! [[ "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --np must be a positive integer" >&2
    exit 2
fi
if ! [[ "$limit" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --limit must be a nonnegative integer" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
runner_local="${script_dir}/run_eirene_training_case.csh"
runner_remote="${remote_root}/run_eirene_training_case.csh"

mkdir -p "$output_root"

ssh "$remote_host" \
    "mkdir -p '$remote_root' && { test -e '$remote_root/baserun' || ln -s '$remote_base' '$remote_root/baserun'; }"
rsync -av "$runner_local" "${remote_host}:${runner_remote}"

processed=0
for case_name in "${cases[@]}"; do
    if (( limit > 0 && processed >= limit )); then
        break
    fi
    if [[ ! "$case_name" =~ ^run_[A-Za-z0-9_-]+$ ]]; then
        echo "ERROR: unsafe case directory name: $case_name" >&2
        exit 2
    fi

    source_case="${source_root}/${case_name}"
    output_case="${output_root}/${case_name}"
    remote_case="${remote_root}/${case_name}"

    if [[ ! -d "$source_case" ]]; then
        echo "ERROR: source case is missing: $source_case" >&2
        exit 2
    fi

    if [[ -f "${output_case}/EIRENE_TRAINING_SUCCESS" ]]; then
        if (cd "$output_case" && shasum -a 256 -c eirene_training_v3.sha256 >/dev/null); then
            echo "SKIP verified: $case_name"
            processed=$((processed + 1))
            continue
        fi
        echo "ERROR: completion marker exists but checksums fail: $output_case" >&2
        exit 4
    fi

    echo "=== $case_name ==="

    remote_ready=0
    if ssh "$remote_host" \
        "cd '$remote_case' 2>/dev/null && test -s eirene_training_v3_b2call_00000000_single_call_0001.nc && test -s eirene_training_v3_b2call_00000000_single_call_0002.nc && test -s eirene_training_v3.sha256 && sha256sum -c eirene_training_v3.sha256 >/dev/null 2>&1"; then
        remote_ready=1
        echo "Using validated products already present on Mora"
    fi

    if (( remote_ready == 0 )); then
        if ssh "$remote_host" "test -e '$remote_case'"; then
            echo "ERROR: incomplete remote working directory already exists: $remote_case" >&2
            echo "Inspect or move it before retrying; it will not be overwritten." >&2
            exit 5
        fi

        ssh "$remote_host" "mkdir '$remote_case'"
        rsync -av \
          --exclude='run_*/' \
          --exclude='balance.nc*' \
          --exclude='b2time.nc*' \
          --exclude='b2batch.nc*' \
          --exclude='b2tallies.nc*' \
          --exclude='b2fmovie*' \
          --exclude='output*' \
          --exclude='*.log' \
          --exclude='b2mn.exe.dir/' \
          "$source_case/" "${remote_host}:${remote_case}/"

        ssh "$remote_host" "tcsh '$runner_remote' '$remote_case' '$mpi_ranks'"
    fi

    mkdir -p "$output_case"
    rsync -av --prune-empty-dirs \
      --include='/eirene_training_v3_*.nc' \
      --include='/eirene_training_validation.log' \
      --include='/eirene_training_v3.sha256' \
      --include='/eirene_training_run.log' \
      --include='/eirene_training_make_dryrun.log' \
      --include='/eirene_training_provenance.txt' \
      --include='/b2mn.dat' \
      --exclude='*' \
      "${remote_host}:${remote_case}/" "$output_case/"

    if [[ -s "${source_case}/params.json" ]]; then
        cp -p "${source_case}/params.json" "${output_case}/source_params.json"
    fi

    file_count="$(find "$output_case" -maxdepth 1 -type f -name 'eirene_training_v3_*.nc' | wc -l | tr -d ' ')"
    if [[ "$file_count" != "2" ]]; then
        echo "ERROR: expected two local NetCDF files, found $file_count" >&2
        exit 6
    fi
    (cd "$output_case" && shasum -a 256 -c eirene_training_v3.sha256)
    touch "${output_case}/EIRENE_TRAINING_SUCCESS"

    if (( keep_remote == 0 )); then
        ssh "$remote_host" \
            "test '$remote_case' != '$remote_root' && rm -rf -- '$remote_case'"
    fi

    echo "PASS archived: $output_case"
    processed=$((processed + 1))
done

echo "PASS processed $processed case(s)"
