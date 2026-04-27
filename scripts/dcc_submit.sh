#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${1:-pipeline}"
CONFIG_PATH="${2:-configs/dcc_gpu.json}"
SCRIPT_PATH="jobs/${JOB_NAME}.sbatch"

export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export CONFIG_PATH

if [[ "$JOB_NAME" == "pipeline" ]]; then
  STAGES=(render-batch convert sanity-check train evaluate)
  PREV_JOB_ID=""

  for STAGE in "${STAGES[@]}"; do
    STAGE_SCRIPT="jobs/${STAGE}.sbatch"
    if [[ ! -f "$STAGE_SCRIPT" ]]; then
      echo "Missing job script: $STAGE_SCRIPT" >&2
      exit 1
    fi

    echo "Submitting $STAGE_SCRIPT with CONFIG_PATH=$CONFIG_PATH"
    if [[ -n "$PREV_JOB_ID" ]]; then
      JOB_ID="$(sbatch --parsable --dependency=afterok:${PREV_JOB_ID} "$STAGE_SCRIPT")"
    else
      JOB_ID="$(sbatch --parsable "$STAGE_SCRIPT")"
    fi
    echo "  submitted ${STAGE} as job ${JOB_ID}"
    PREV_JOB_ID="$JOB_ID"
  done

  echo "${JOB_NAME} submission chain complete. Final job id: ${PREV_JOB_ID}"
  exit 0
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Missing job script: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Submitting $SCRIPT_PATH with CONFIG_PATH=$CONFIG_PATH"
sbatch "$SCRIPT_PATH"
