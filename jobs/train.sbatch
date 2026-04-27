#!/bin/bash
#SBATCH --job-name=sta561-pest-train
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dcc/train-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-configs/dcc_gpu.json}"

cd "$PROJECT_ROOT"

uv sync
echo "Running train job with CONFIG_PATH=$CONFIG_PATH"
uv run pest-pipeline train --config "$CONFIG_PATH"
