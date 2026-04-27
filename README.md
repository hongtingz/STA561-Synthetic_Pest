# Prob ML Pest Pipeline

End-to-end STA 561 pest-detection pipeline organized around the A+ target:

1. Take a kitchen photo as input
2. Generate labeled synthetic pest video in Blender
3. Export frame-level annotations
4. Train transformer and detector baselines
5. Run the pipeline reproducibly on the Duke compute cluster

## Design Goals

- `src/` package layout instead of loose top-level scripts
- `uv` as the source of truth for dependency and environment management
- one CLI entrypoint for local work and DCC submission
- configuration-first workflow via JSON config files
- clear separation between render, dataset, training, inference, and cluster orchestration

## Current State

The repository now supports an end-to-end workflow rather than a pure scaffold.
The package structure, CLI, DCC helpers, and configuration flow support
rendering, dataset conversion, sanity checking, detector training, evaluation,
and inference in one reproducible workflow.

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

Model training and inference now use a transformer-style detector path by
default: `vit` maps to the Hugging Face `hustvl/yolos-tiny` object detector.
The same training, evaluation, and inference entrypoints also support a
torchvision Faster R-CNN fallback, and the converter writes YOLO labels for an
optional Ultralytics comparison. Dataset sanity checking and checkpoint
evaluation produce structured reports and visual galleries for review.

The current repo provides:

- stable module boundaries
- consistent config loading
- directory conventions
- DCC job wiring

## Detector Path

The implemented detector progression is:

```text
kitchen photo
-> Blender render-batch
-> frame annotations
-> COCO export
-> sanity-check
-> ViT/YOLOS-tiny detector training
-> threshold selection
-> evaluation on validation + neg_test
```

The DCC configs default to `detector_model: "vit"`, which resolves to
`hustvl/yolos-tiny`. Faster R-CNN remains available through
`detector_model: "fasterrcnn_mobilenet_v3_large_320_fpn"` for a lightweight
engineering baseline, and `pest-pipeline train-yolo` remains available as an
optional YOLO comparison path.

## Quick Start

```bash
uv sync
uv run pest-pipeline doctor --config configs/dcc_gpu.json
uv run pest-pipeline plan --config configs/dcc_gpu.json
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job render-batch
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job train
```

`configs/dcc_gpu.json` is the main instructor-facing and DCC-oriented
configuration. It switches Blender to GPU rendering with the `CUDA` backend and
uses the production cluster resource profile. `configs/base.json` remains
available for local CPU development, and `configs/dcc_gpu_smoke.json` is kept
only for short validation runs.

For the instructor-facing reproduction and deployment description, see
[DCC_DEPLOYMENT.md](DCC_DEPLOYMENT.md).

For the original course project description in Markdown form, see
[PROJECT_SPEC.md](PROJECT_SPEC.md).

For a lightweight instructor-facing notebook that reads existing DCC artifacts,
see [dcc_pipeline_demo.ipynb](notebooks/dcc_pipeline_demo.ipynb).

## Repository Layout

```text
assets/           Local third-party 3D assets and attribution notes
configs/          Runtime configuration files
jobs/             DCC / Slurm job scripts
scripts/          Small operational shell helpers
src/prob_ml/      Python package
tests/            Lightweight tests for the project modules
```

## Data Conventions

- `data/inputs/`
  Single-image local experiments and quick checks.
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
  Trains the configured detector on exported COCO data. The DCC config defaults
  to `vit` / `hustvl/yolos-tiny`; Faster R-CNN remains available by changing
  `training.detector_model`.
- `pest-pipeline evaluate --config ...`
  Loads a trained detector checkpoint and writes a unified validation/negative
  holdout report plus false-positive/false-negative example visualizations.
- `pest-pipeline train-yolo --config ...`
  Optionally trains a YOLO detector from `artifacts/dataset/yolo/data.yaml`.
  This requires installing `ultralytics`; the built-in `train` path does not
  require it.
- `pest-pipeline infer --config ...`
  Loads a detector checkpoint, runs single-image inference, and writes a
  visualization plus JSON predictions.
- `pest-pipeline pipeline --config ...`
  Runs the full local pipeline sequence in one process.
- `pest-pipeline dcc-submit --config ... --job pipeline`
  Submits a Slurm job. The helper script `scripts/dcc_submit.sh pipeline
  configs/dcc_gpu.json` chains `render-batch`, `convert`, `sanity-check`,
  `train`, and `evaluate` with Slurm dependencies.

## DCC Direction

The cluster-facing workflow is designed around:

- config-driven jobs
- reproducible output directories under `artifacts/`
- Slurm entrypoints under `jobs/`
- a single Python package that can be called locally and on DCC

## Notebook Demo

The repository includes a lightweight DCC-oriented notebook demo:

- [dcc_pipeline_demo.ipynb](notebooks/dcc_pipeline_demo.ipynb)

It is intended to be opened after `render-batch`, `convert`, `sanity-check`,
`train`, and `evaluate` have already produced artifacts. The notebook does not
launch expensive jobs; it reads existing outputs and displays:

- one input kitchen image
- one rendered frame with stored pest bounding boxes
- dataset summary artifacts
- evaluation metrics and example failure cases

## Active Result Updates

The repo is structured so the latest DCC artifacts can be dropped into
`artifacts/` without code changes. The active update path is:

1. Complete the DCC render and detector runs.
2. Confirm `dataset_summary.json`, `training_report.json`, and
   `detector_evaluation_report.json`.
3. Use the notebook and report artifacts to refresh the final qualitative
   examples and quantitative tables.
