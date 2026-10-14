#!/usr/bin/env bash
# Project: SOLPS-ITER/EIRENE training-data generation for SOLSTICE
# Author: Abdou Diaw, Oak Ridge National Laboratory
#
# Mora-side resumable controller. It downloads one archived case from
# Dropbox, runs one B2/EIRENE step, uploads and verifies only the training
# products, removes the successful temporary case, and then proceeds.

set -euo pipefail

dropbox_source="dropbox:SOLPS_DB/ens__DIII-D__APP-FPP__D_C_Ne__ss__lhs__20250124_144649"
dropbox_output="dropbox:SOLPS_DB/eirene_training_v3__ens__DIII-D__APP-FPP__D_C_Ne__ss__lhs__20250124_144649"
work_root="/home/cloud/solps-runs/diii-d/eirene_training_campaign_v3"
base_run="/home/cloud/solps-runs/diii-d/baserun"
mpi_ranks=64
run_all=0
limit=0
cases=()
skip_cases=()

usage() {
    cat <<'EOF'
Usage:
  generate_eirene_training_db_cloud.sh --case RUN_NAME [options]
  generate_eirene_training_db_cloud.sh --all [--limit N] [options]

Options:
  --case RUN_NAME       Process one run_* directory; may be repeated.
  --all                 Process every top-level run_* directory in Dropbox.
  --skip-case RUN_NAME  Skip a known unusable run_* directory; may be repeated.
  --limit N             Process at most N selected cases.
  --np N                MPI ranks per case (default: 64).
  --source REMOTE_PATH  Override the Dropbox source path.
  --output REMOTE_PATH  Override the Dropbox output path.
  -h, --help            Show this help.
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
        --skip-case)
            skip_cases+=("$2")
            shift 2
            ;;
        --limit)
            limit="$2"
            shift 2
            ;;
        --np)
            mpi_ranks="$2"
            shift 2
            ;;
        --source)
            dropbox_source="${2%/}"
            shift 2
            ;;
        --output)
            dropbox_output="${2%/}"
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
if ! [[ "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --np must be a positive integer" >&2
    exit 2
fi
if ! [[ "$limit" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --limit must be a nonnegative integer" >&2
    exit 2
fi
for skip_case in "${skip_cases[@]}"; do
    if [[ ! "$skip_case" =~ ^run_[A-Za-z0-9_-]+$ ]]; then
        echo "ERROR: unsafe --skip-case value: $skip_case" >&2
        exit 2
    fi
done

command -v rclone >/dev/null || { echo "ERROR: rclone is unavailable" >&2; exit 2; }
command -v tcsh >/dev/null || { echo "ERROR: tcsh is unavailable" >&2; exit 2; }

runner="${work_root}/run_eirene_training_case.csh"
campaign_log="${work_root}/eirene_training_cloud_campaign.log"

mkdir -p "$work_root"
if [[ ! -e "${work_root}/baserun" ]]; then
    ln -s "$base_run" "${work_root}/baserun"
fi
[[ -d "${work_root}/baserun" ]] || { echo "ERROR: invalid baserun" >&2; exit 2; }
[[ -s "$runner" ]] || { echo "ERROR: Mora runner is missing: $runner" >&2; exit 2; }

if (( run_all == 1 )); then
    while IFS= read -r case_name; do
        case_name="${case_name%/}"
        [[ "$case_name" == run_* ]] && cases+=("$case_name")
    done < <(rclone lsf "$dropbox_source" --dirs-only --max-depth 1 | sort)
fi

if (( ${#cases[@]} == 0 )); then
    echo "ERROR: no cases selected or Dropbox source path is incorrect" >&2
    exit 2
fi

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$campaign_log"
}

processed=0
skipped_missing=0
skipped_requested=0
for case_name in "${cases[@]}"; do
    if (( limit > 0 && processed >= limit )); then
        break
    fi
    if [[ ! "$case_name" =~ ^run_[A-Za-z0-9_-]+$ ]]; then
        log "ERROR unsafe case directory name: $case_name"
        exit 2
    fi

    skip_requested=0
    for skip_case in "${skip_cases[@]}"; do
        if [[ "$case_name" == "$skip_case" ]]; then
            skip_requested=1
            break
        fi
    done
    if (( skip_requested == 1 )); then
        log "SKIP requested known failure ${case_name}"
        processed=$((processed + 1))
        skipped_requested=$((skipped_requested + 1))
        continue
    fi

    source_case="${dropbox_source}/${case_name}"
    output_case="${dropbox_output}/${case_name}"
    work_case="${work_root}/${case_name}"

    if rclone lsf "$output_case" --files-only --include='EIRENE_TRAINING_SUCCESS' 2>/dev/null | grep -q '^EIRENE_TRAINING_SUCCESS$'; then
        log "SKIP already archived ${case_name}"
        processed=$((processed + 1))
        continue
    fi

    log "START ${case_name}"

    if [[ -d "$work_case" ]]; then
        if [[ -f "${work_case}/EIRENE_TRAINING_SUCCESS" ]] && \
           (cd "$work_case" && sha256sum -c eirene_training_v3.sha256 >/dev/null); then
            log "REUSE validated Mora products ${case_name}"
        else
            log "ERROR incomplete Mora working directory retained at ${work_case}"
            exit 5
        fi
    else
        mkdir "$work_case"
        if ! rclone copy "$source_case" "$work_case" \
          --exclude='/run_*/**' \
          --exclude='/balance.nc*' \
          --exclude='/b2time.nc*' \
          --exclude='/b2batch.nc*' \
          --exclude='/b2tallies.nc*' \
          --exclude='/b2fmovie*' \
          --exclude='/output*' \
          --exclude='/*.log' \
          --exclude='/b2mn.exe.dir/**' \
          --links; then
            log "ERROR Dropbox download failed; retained ${work_case}"
            exit 6
        fi

        if tcsh "$runner" "$work_case" "$mpi_ranks"; then
            runner_status=0
        else
            runner_status=$?
        fi
        if (( runner_status == 20 )); then
            log "SKIP missing required case input ${case_name}"
            rm -rf -- "$work_case"
            processed=$((processed + 1))
            skipped_missing=$((skipped_missing + 1))
            continue
        elif (( runner_status != 0 )); then
            log "ERROR SOLPS/EIRENE run failed; retained ${work_case}"
            exit 7
        fi
    fi

    if [[ -s "${work_case}/params.json" ]]; then
        cp -p "${work_case}/params.json" "${work_case}/source_params.json"
    fi

    if ! rclone copy "$work_case" "$output_case" \
      --include='/eirene_training_v3_*.nc' \
      --include='/eirene_training_validation.log' \
      --include='/eirene_training_v3.sha256' \
      --include='/eirene_training_run.log' \
      --include='/eirene_training_make_dryrun.log' \
      --include='/eirene_training_setup_*.log' \
      --include='/eirene_training_provenance.txt' \
      --include='/source_params.json' \
      --include='/b2mn.dat' \
      --include='/EIRENE_TRAINING_SUCCESS'; then
        log "ERROR Dropbox upload failed; retained ${work_case}"
        exit 8
    fi

    if ! rclone check "$work_case" "$output_case" --one-way \
      --include='/eirene_training_v3_*.nc' \
      --include='/eirene_training_validation.log' \
      --include='/eirene_training_v3.sha256' \
      --include='/eirene_training_run.log' \
      --include='/eirene_training_make_dryrun.log' \
      --include='/eirene_training_setup_*.log' \
      --include='/eirene_training_provenance.txt' \
      --include='/source_params.json' \
      --include='/b2mn.dat' \
      --include='/EIRENE_TRAINING_SUCCESS'; then
        log "ERROR Dropbox verification failed; retained ${work_case}"
        exit 9
    fi

    rm -rf -- "$work_case"
    log "PASS archived and removed Mora copy ${case_name}"
    processed=$((processed + 1))

    rclone copyto "$campaign_log" "${dropbox_output}/eirene_training_cloud_campaign.log" || true
done

log "PASS processed ${processed} case(s); skipped ${skipped_missing} with missing required input and ${skipped_requested} requested"
rclone copyto "$campaign_log" "${dropbox_output}/eirene_training_cloud_campaign.log"
