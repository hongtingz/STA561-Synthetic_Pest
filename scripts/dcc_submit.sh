#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${1:-pipeline}"
CONFIG_PATH="${2:-configs/base.json}"
SCRIPT_PATH="jobs/${JOB_NAME}.sbatch"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Missing job script: $SCRIPT_PATH" >&2
  exit 1
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export CONFIG_PATH

echo "Submitting $SCRIPT_PATH with CONFIG_PATH=$CONFIG_PATH"
sbatch "$SCRIPT_PATH"
