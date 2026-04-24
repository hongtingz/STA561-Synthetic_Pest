# Prob ML Pest Pipeline

Rebuilt foundation for the STA 561 pest-detection final project, with the repo
organized around the A+ target:

1. Take a kitchen photo as input
2. Generate labeled synthetic pest video in Blender
3. Export frame-level annotations
4. Train a ViT-based model
5. Run the pipeline reproducibly on the Duke compute cluster

## Design Goals

- `src/` package layout instead of loose top-level scripts
- `uv` as the source of truth for dependency and environment management
- one CLI entrypoint for local work and DCC submission
- configuration-first workflow via JSON config files
- clear separation between render, dataset, training, inference, and cluster orchestration

## Current State

This branch is a clean rebuild scaffold. The package structure, CLI, DCC helpers,
and configuration flow are in place so the repo can grow into the full A+ system
without another reorganization pass.

The render and dataset-packaging path now supports:

- manifest-driven batch render preparation
- Blender frame + per-pest JSON annotations from `src/prob_ml/blender/render_scene.py`
- **Multi-pest scenes** via `render.pest_types` (e.g. `mouse`, `rat`, `cockroach`), each
  with a woven path (two intermediate waypoints) in the generated `layout.json`
- **H.264 video** after rendering: with `render.execute: true` and `render.mux_video: true`
  (default), frames under `dataset.frames_dir` are muxed to `dataset.video_output` using
  `ffmpeg` (set `render.mux_video: false` to skip)
- COCO dataset export for detector training
- YOLO dataset export for fast detector baselines
- negative-only real-kitchen holdout packaging for false-positive evaluation

**Optional legacy demo** (`demo/render_demo.py`): self-contained Cycles kitchen + the same
three pest families; can mux MP4 the same way if `ffmpeg` is available. Rebuild from
frames only: `uv run python demo/frames_to_video.py` (expects `artifacts/render/frames/`
and 5-digit `frame_%05d.png` names from the main renderer).

Model training and inference now include a lightweight torchvision Faster R-CNN
baseline with lightweight augmentation, optional pretrained weights, and
threshold-sweep TDR/FPR reporting. The converter also writes YOLO labels, and
`pest-pipeline train-yolo` provides an optional Ultralytics path for an
additional detector baseline. Dataset sanity checking and checkpoint evaluation
now produce report artifacts and visual galleries for final write-up evidence.
Final experiments and result selection are still in progress. The current repo provides:

- stable module boundaries
- consistent config loading
- directory conventions
- DCC job wiring

## Quick Start

```bash
uv sync
uv run pest-pipeline doctor
uv run pest-pipeline plan --config configs/base.json
uv run pest-pipeline convert --config configs/base.json
uv run pest-pipeline dcc-submit --config configs/base.json --job pipeline
```

`configs/base.json` is kept conservative for local smoke tests and uses CPU
rendering. `configs/dcc_gpu.json` is the DCC-oriented variant that switches
Blender to GPU rendering with the `CUDA` backend.

For cluster usage notes and submission examples, see
[DCC_USAGE.md](/Users/hongting/projects/prob_ml/DCC_USAGE.md).

For the original course project description in Markdown form, see
[PROJECT_SPEC.md](/Users/hongting/projects/prob_ml/PROJECT_SPEC.md).

## Repository Layout

```text
assets/           Local third-party 3D assets and attribution notes
configs/          Runtime configuration files
jobs/             DCC / Slurm job scripts
scripts/          Small operational shell helpers
src/prob_ml/      Python package
tests/            Lightweight tests for the scaffold
```

## Data Conventions

- `data/inputs/`
  Single-image local experiments and smoke tests.
- `data/raw/kitchen/images/`
  External real kitchen photo corpus in its original form, used to drive large-batch synthetic generation.
- `data/raw/kitchen/metadata/manifest.csv`
  Optional manifest for tracking split membership, source, or quality flags.
- `data/intermediate/`
  Derived but non-final assets such as layout specs and cached preprocessing outputs.
- `artifacts/`
  Generated outputs only: rendered frames, converted annotations, trained models, previews.
- `assets/pests/`
  Optional downloaded pest models. Large model and texture binaries are ignored by
  git; keep attribution in `assets/pests/CREDITS.md`.

Key dataset outputs produced by `pest-pipeline convert`:

- `artifacts/dataset/coco_annotations.json`
  Combined positive rendered dataset.
- `artifacts/dataset/coco_train.json`
  COCO training split for positive rendered data.
- `artifacts/dataset/coco_val.json`
  COCO validation split for positive rendered data.
- `artifacts/dataset/coco_neg_test.json`
  COCO-style negative-only real kitchen holdout.
- `artifacts/dataset/neg_test_images.json`
  Real kitchen negative-only image listing for false-positive evaluation.
- `artifacts/dataset/yolo/`
  YOLO images, labels, and `data.yaml` derived from the same split definitions.
- `artifacts/dataset/dataset_summary.json`
  Split counts, category mappings, and missing-background summary.

Current split interpretation:

- `train` / `val`
  Positive rendered or composited pest data.
- `neg_test`
  Real kitchen images with no pests, used for false-positive evaluation.

Because the team does not currently have real positive pest images, any future
positive `test` split must come from held-out rendered or composited data.

## CLI Commands

- `pest-pipeline doctor`
  Checks for expected local tools and directories.
- `pest-pipeline plan --config configs/base.json`
  Prints the resolved pipeline plan and output locations.
- `pest-pipeline render --config ...`
  Reserved for Blender-based synthetic data generation.
- `pest-pipeline render-batch --config ...`
  Runs the manifest-driven batch render entrypoint over kitchen photos.
- `pest-pipeline convert --config ...`
  Validates rendered annotations and exports COCO + YOLO datasets together with
  a negative-only real-kitchen holdout manifest.
- `pest-pipeline sanity-check --config ...`
  Validates COCO split integrity, bbox bounds, class counts, split leakage, and
  writes annotation overlay images under `artifacts/reports/`.
- `pest-pipeline train --config ...`
  Trains the built-in Faster R-CNN detector baseline on exported COCO data and
  writes a checkpoint plus `training_report.json`, including threshold-sweep
  summaries for TDR/FPR selection.
- `pest-pipeline evaluate --config ...`
  Loads a trained detector checkpoint and writes a unified validation/negative
  holdout report plus false-positive/false-negative example visualizations.
- `pest-pipeline train-yolo --config ...`
  Optionally trains a YOLO detector from `artifacts/dataset/yolo/data.yaml`.
  This requires installing `ultralytics`; the built-in Faster R-CNN path does
  not require it.
- `pest-pipeline infer --config ...`
  Loads a detector checkpoint, runs single-image inference, and writes a
  visualization plus JSON predictions.
- `pest-pipeline pipeline --config ...`
  Reserved for the full end-to-end local pipeline.
- `pest-pipeline dcc-submit --config ... --job pipeline`
  Prints the `sbatch` command for DCC execution. Use `--job train-yolo` for
  the optional YOLO baseline.

## DCC Direction

The cluster-facing workflow is designed around:

- config-driven jobs
- reproducible output directories under `artifacts/`
- Slurm entrypoints under `jobs/`
- a single Python package that can be called locally and on DCC

## Next Build Steps

1. Scale batch rendering and confirm output quality across more kitchen backgrounds
2. Run `sanity-check` before training and fix any dataset/report errors
3. Run smoke training on DCC and confirm checkpoint/report artifacts
4. Run `evaluate` plus the optional YOLO baseline if dependencies are available
5. Tighten end-to-end docs and final metric tables
