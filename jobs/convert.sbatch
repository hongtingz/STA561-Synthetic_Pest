#!/bin/bash
#SBATCH --job-name=sta561-pest-convert
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dcc/convert-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-configs/dcc_gpu.json}"

cd "$PROJECT_ROOT"

uv sync
echo "Running dataset conversion with CONFIG_PATH=$CONFIG_PATH"
uv run pest-pipeline convert --config "$CONFIG_PATH"
