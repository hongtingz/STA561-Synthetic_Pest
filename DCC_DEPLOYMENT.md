# DCC Deployment And Reproduction Guide

This document is written for an instructor or reviewer who wants to understand
how this project is deployed on the Duke Compute Cluster (DCC) and how to
reproduce the main pipeline stages.

## 1. Purpose

The project takes kitchen images as input, generates synthetic pest scenes in
Blender, converts those renders into detection datasets, and then trains and
evaluates an object detector. DCC is used for reproducible batch rendering and
GPU-based detector training.

The project currently supports this workflow:

```text
render-batch -> convert -> sanity-check -> train -> evaluate -> infer
```

The current `train` job targets the configured detector in `configs/dcc_gpu.json`.
The main DCC profile uses `detector_model: "vit"`, which resolves to the
Hugging Face `hustvl/yolos-tiny` object detector. The same data export,
Slurm structure, and evaluation reports also support the torchvision
Faster R-CNN fallback and the optional YOLO comparison path.

## 2. DCC Environment Assumptions

The provided Slurm scripts are configured for the following DCC resources:

- Partition: `scavenger-h200`
- Account: `scavenger-h200`
- GPU: `gpu:h200:1` for render/train/evaluate jobs
- CPU-only sanity-check job

The project expects:

- `uv` for Python environment management
- `blender` available on the cluster `PATH`
- the repository present on the cluster filesystem
- the input kitchen dataset available under the repository's `data/` path

## 3. Repository Layout Relevant To DCC

Important directories:

```text
configs/           runtime JSON configs
jobs/              Slurm job scripts
scripts/           helper shell scripts for submission
src/prob_ml/       pipeline implementation
data/              input images and manifests
assets/pests/      optional local 3D pest assets
artifacts/         generated outputs
logs/dcc/          Slurm logs
```

Important configs:

- `configs/base.json`
  Local development config.
- `configs/dcc_gpu.json`
  Main DCC config for formal rendering, training, and evaluation runs.
- `configs/dcc_gpu_smoke.json`
  Optional short validation config used only for quick cluster checks.

## 4. One-Time Setup

Clone the repository and enter the project directory:

```bash
git clone git@github.com:hongtingz/prob_ml.git
cd prob_ml
```

Confirm tools:

```bash
uv run pest-pipeline doctor --config configs/dcc_gpu.json
```

Confirm config resolution:

```bash
uv run pest-pipeline plan --config configs/dcc_gpu.json
```

If the project uses external downloaded 3D pest assets, place them under:

```text
assets/pests/rodent/scary_ratmouse/
assets/pests/cockroach/ck_cockroach/
```

The current default rendering style is `hybrid`:

- `mouse` and `rat` use procedural pests by default
- `cockroach` uses the downloaded asset when available
- missing assets fall back to procedural rendering rather than failing

## 5. Main Slurm Jobs

The following batch jobs are provided:

- `jobs/render.sbatch`
- `jobs/render-batch.sbatch`
- `jobs/sanity-check.sbatch`
- `jobs/train.sbatch`
- `jobs/evaluate.sbatch`
- `jobs/train-yolo.sbatch`
- `jobs/pipeline.sbatch`

Each script:

1. changes into the project root
2. runs `uv sync`
3. executes the corresponding `pest-pipeline` command

## 6. Recommended Reproduction Sequence

### Step 1: Validate Rendering On DCC

```bash
bash scripts/dcc_submit.sh render-batch configs/dcc_gpu.json
```

This stage generates:

- rendered frames
- per-frame annotation JSON
- layout JSON
- layout diagnostics and preview files

### Step 2: Convert To Training Dataset

```bash
uv run pest-pipeline convert --config configs/dcc_gpu.json
```

This stage writes:

- COCO detection annotations
- YOLO labels and `data.yaml`
- split summaries

### Step 3: Run Dataset Sanity Checks

```bash
bash scripts/dcc_submit.sh sanity-check configs/dcc_gpu.json
```

This stage validates:

- bbox format
- file existence
- category counts
- split assumptions
- negative-only split integrity

### Step 4: Train The Main Detector

```bash
bash scripts/dcc_submit.sh train configs/dcc_gpu.json
```

The main DCC detector path is the transformer-style `vit` configuration
(`hustvl/yolos-tiny`). Training writes epoch checkpoints, a final
`detector.pt`, and `training_report.json`. Faster R-CNN remains available by
changing `training.detector_model` to
`fasterrcnn_mobilenet_v3_large_320_fpn`.

### Step 5: Evaluate The Trained Detector

```bash
bash scripts/dcc_submit.sh evaluate configs/dcc_gpu.json
```

This stage writes:

- threshold-based evaluation report
- false-positive examples
- false-negative examples

### Step 6: Run Inference

Inference can be run after training on a selected image:

```bash
uv run pest-pipeline infer --config configs/dcc_gpu.json
```

After `render-batch`, `convert`, `sanity-check`, `train`, and `evaluate` have
finished, the generated artifacts can also be reviewed through the notebook:

- [dcc_pipeline_demo.ipynb](notebooks/dcc_pipeline_demo.ipynb)

This notebook is intentionally read-only with respect to expensive pipeline
stages. It reads existing outputs from `artifacts/` and is suitable for an
instructor-facing demonstration on DCC or code-server.

Optional short validation run:

```bash
uv run pest-pipeline doctor --config configs/dcc_gpu_smoke.json
bash scripts/dcc_submit.sh render-batch configs/dcc_gpu_smoke.json
```

## 7. Monitoring And Logs

Check live queue state:

```bash
squeue -u "$USER"
```

Useful logs:

```bash
tail -f logs/dcc/render-batch-<jobid>.out
tail -f logs/dcc/sanity-check-<jobid>.out
tail -f logs/dcc/train-<jobid>.out
tail -f logs/dcc/evaluate-<jobid>.out
```

Check final job state:

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode
```

## 8. Expected Output Artifacts

Synthetic render outputs:

- `artifacts/batch_render*/kitchen_*/frames/`
- `artifacts/batch_render*/kitchen_*/annotations.json`
- `artifacts/batch_render*/kitchen_*/layout.json`

Dataset outputs:

- `artifacts/dataset/coco_train.json`
- `artifacts/dataset/coco_val.json`
- `artifacts/dataset/coco_neg_test.json`
- `artifacts/dataset/yolo/data.yaml`

Sanity-check outputs:

- `artifacts/reports/dataset_sanity_report.json`
- `artifacts/reports/sanity_overlays/`

Training outputs:

- `artifacts/models/detector/detector.pt`
- `artifacts/models/detector/training_report.json`

Evaluation outputs:

- `artifacts/reports/evaluation/detector_evaluation_report.json`
- `artifacts/reports/evaluation/failure_examples/`

Inference outputs:

- `artifacts/infer/predictions.json`
- `artifacts/infer/infer_result.jpg`

Notebook demo:

- `notebooks/dcc_pipeline_demo.ipynb`

## 9. Reproducibility Notes

- All runtime choices are config-driven through JSON files in `configs/`.
- Slurm jobs call the same Python CLI used locally.
- Generated artifacts remain outside version control.
- Optional external pest assets are not committed to git and must be placed on
  DCC separately if that visual path is desired.
- If those assets are missing, the pipeline still runs through procedural
  fallback logic.
- The recommended formal reproduction path for review uses
  `configs/dcc_gpu.json`; the smoke config is intentionally smaller and is not
  the primary submission path.

## 10. Current Scope And Limitations

- The project supports detector training and evaluation, but result quality still
  depends on the realism and diversity of synthetic renders.
- Negative-only real kitchen images are used to estimate false positives.
- Positive evaluation data is still synthetic or synthetic-derived unless future
  real labeled pest imagery is added.
- The main documented training path is the `vit` / `hustvl/yolos-tiny`
  detector in `configs/dcc_gpu.json`.
- YOLO support exists as an optional comparison path, and Faster R-CNN remains a
  lightweight fallback.

## 11. Detector Configuration

The main detector stage is:

```text
render-batch
-> convert
-> sanity-check
-> ViT/YOLOS-tiny detector training
-> threshold sweep
-> evaluation
```

The intent is to keep rendering, annotation export, dataset packaging, and DCC
submission unchanged while allowing detector swaps through JSON config. This
lets the team compare ViT/YOLOS-tiny, Faster R-CNN, and optional YOLO without
changing the upstream data-generation pipeline.
