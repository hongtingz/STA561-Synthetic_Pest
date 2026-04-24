#!/bin/bash
#SBATCH --job-name=sta561-pest-yolo
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dcc/train-yolo-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-configs/dcc_gpu.json}"

cd "$PROJECT_ROOT"

uv sync
echo "Running optional YOLO job with CONFIG_PATH=$CONFIG_PATH"
uv run --with ultralytics pest-pipeline train-yolo --config "$CONFIG_PATH"
