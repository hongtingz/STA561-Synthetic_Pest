#!/bin/bash
#SBATCH --job-name=sta561-pest-render-batch
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dcc/render-batch-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-configs/dcc_gpu.json}"

cd "$PROJECT_ROOT"

uv sync
echo "Running render-batch job with CONFIG_PATH=$CONFIG_PATH"
uv run pest-pipeline render-batch --config "$CONFIG_PATH"
