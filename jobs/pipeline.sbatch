#!/bin/bash
#SBATCH --job-name=sta561-pest-pipeline
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dcc/pipeline-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-configs/dcc_gpu.json}"

cd "$PROJECT_ROOT"

uv sync
echo "Running pipeline job with CONFIG_PATH=$CONFIG_PATH"
uv run pest-pipeline pipeline --config "$CONFIG_PATH"
