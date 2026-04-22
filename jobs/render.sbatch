#!/bin/bash
#SBATCH --job-name=sta561-pest-render
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dcc/render-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-configs/base.json}"

cd "$PROJECT_ROOT"

uv sync
uv run pest-pipeline render --config "$CONFIG_PATH"
