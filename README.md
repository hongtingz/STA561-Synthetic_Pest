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

The Blender generation, dataset conversion, model training, and inference modules
are intentionally scaffolded as structured placeholders right now. They provide:

- stable module boundaries
- consistent config loading
- directory conventions
- DCC job wiring

## Quick Start

```bash
uv sync
uv run pest-pipeline doctor
uv run pest-pipeline plan --config configs/base.json
uv run pest-pipeline dcc-submit --config configs/base.json --job pipeline
```

`configs/base.json` is kept conservative for local smoke tests and uses CPU
rendering. `configs/dcc_gpu.json` is the DCC-oriented variant that switches
Blender to GPU rendering with the `CUDA` backend.

For cluster usage notes and submission examples, see
[docs/dcc_usage.md](/Users/hongting/projects/prob_ml/docs/dcc_usage.md).

For the original course project description in Markdown form, see
[docs/project_spec.md](/Users/hongting/projects/prob_ml/docs/project_spec.md).

## Repository Layout

```text
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

## CLI Commands

- `pest-pipeline doctor`
  Checks for expected local tools and directories.
- `pest-pipeline plan --config configs/base.json`
  Prints the resolved pipeline plan and output locations.
- `pest-pipeline render --config ...`
  Reserved for Blender-based synthetic data generation.
- `pest-pipeline convert --config ...`
  Reserved for annotation conversion and dataset packaging.
- `pest-pipeline train --config ...`
  Reserved for model training.
- `pest-pipeline infer --config ...`
  Reserved for inference and evaluation.
- `pest-pipeline pipeline --config ...`
  Reserved for the full end-to-end local pipeline.
- `pest-pipeline dcc-submit --config ... --job pipeline`
  Prints the `sbatch` command for DCC execution.

## DCC Direction

The cluster-facing workflow is designed around:

- config-driven jobs
- reproducible output directories under `artifacts/`
- Slurm entrypoints under `jobs/`
- a single Python package that can be called locally and on DCC

## Next Build Steps

1. Implement Blender scene generation from a kitchen-photo-derived layout spec
2. Add multi-pest assets and animation logic
3. Implement COCO/video dataset packaging
4. Wire in ViT training and evaluation
5. Add instructor-style holdout evaluation and DCC batch jobs
