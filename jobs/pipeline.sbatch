#!/bin/bash
#SBATCH --job-name=sta561-pest-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dcc/pipeline-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-configs/base.json}"

cd "$PROJECT_ROOT"

uv sync
uv run pest-pipeline pipeline --config "$CONFIG_PATH"
