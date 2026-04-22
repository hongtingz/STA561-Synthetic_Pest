# DCC Usage Guide

This project keeps two main runtime configs:

- `configs/base.json`
  Local development and smoke-test config. It keeps Blender on `CPU` so local
  debugging stays predictable.
- `configs/dcc_gpu.json`
  DCC-oriented config. It switches Blender to `GPU` with `CUDA`, which is the
  config to start from on H200-class nodes.

## Before You Submit

1. Make sure the repo is on DCC and you are in the project root:

```bash
cd /path/to/prob_ml
```

2. Confirm the config resolves correctly:

```bash
uv run pest-pipeline plan --config configs/dcc_gpu.json
```

3. If you only want a quick validation run, keep the job small and test a
single image or a tiny subset first. The current pipeline is still closer to a
single-image smoke flow than a full batch render array.

## Recommended Submission Commands

Use the Python CLI if you want the generated `sbatch` command:

```bash
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job render
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job train
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job pipeline
```

Use the helper shell script if you want direct submission:

```bash
bash scripts/dcc_submit.sh render configs/dcc_gpu.json
bash scripts/dcc_submit.sh train configs/dcc_gpu.json
bash scripts/dcc_submit.sh pipeline configs/dcc_gpu.json
```

The helper defaults to `configs/dcc_gpu.json`, so this also works:

```bash
bash scripts/dcc_submit.sh render
```

## What The Job Scripts Do

- `jobs/render.sbatch`
  Runs `uv run pest-pipeline render --config ...`
- `jobs/train.sbatch`
  Runs `uv run pest-pipeline train --config ...`
- `jobs/pipeline.sbatch`
  Runs the whole local pipeline entrypoint in one Slurm job

All three job scripts now default to `configs/dcc_gpu.json`.

## Logs

Slurm logs are written to:

- `logs/dcc/render-<jobid>.out`
- `logs/dcc/train-<jobid>.out`
- `logs/dcc/pipeline-<jobid>.out`

Useful commands after submission:

```bash
squeue -u "$USER"
tail -f logs/dcc/render-<jobid>.out
tail -f logs/dcc/train-<jobid>.out
```

## Current Caveats

- The current `render` stage is validated for smoke tests and small runs. It is
  not yet a manifest-driven batch renderer over all kitchen images.
- The current `train` and `infer` stages are still scaffolds, so DCC is most
  useful right now for render experiments and environment validation.
- Full-scale dataset generation will likely need job arrays and longer Slurm
  time limits than the current starter scripts.

## Suggested DCC Workflow Right Now

1. Validate the config:

```bash
uv run pest-pipeline plan --config configs/dcc_gpu.json
```

2. Run a small render smoke test:

```bash
bash scripts/dcc_submit.sh render configs/dcc_gpu.json
```

3. Inspect the output log and generated artifacts.

4. Only after that, scale the render stage up and then wire in batch manifest
processing.
