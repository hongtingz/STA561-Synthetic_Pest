# Branch Update: `a-plus-dcc-pipeline`

This document summarizes the new work added on this branch so teammates or a
future AI coding session can continue from the current state quickly.

## High-Level Summary

This branch upgrades the project from a data-generation scaffold into a more
complete detector experiment pipeline.

The main path is now:

```bash
render-batch -> convert -> sanity-check -> train -> evaluate -> infer
```

The built-in detector path uses a lightweight torchvision Faster R-CNN model on
COCO annotations. YOLO is supported as an optional comparison path using the
YOLO labels exported by the converter.

## New Main Capabilities

- COCO and YOLO dataset export are both supported.
- Faster R-CNN detector training is implemented.
- Single-image detector inference is implemented.
- Dataset sanity checking is implemented.
- Post-training checkpoint evaluation is implemented.
- Optional YOLO training is implemented.
- DCC sbatch jobs now cover training, evaluation, sanity checking, and optional
  YOLO training.

## New or Heavily Updated Files

- `src/prob_ml/detector.py`
  Shared detector utilities: COCO dataset loader, bbox matching, TDR/FPR
  summaries, model construction, checkpoint payloads, device selection.

- `src/prob_ml/train.py`
  Built-in Faster R-CNN training. Reads COCO train/val/neg_test files, trains a
  detector, writes `detector.pt`, and writes `training_report.json`.

- `src/prob_ml/infer.py`
  Single-image inference from a trained detector checkpoint. Writes prediction
  JSON and a visualization image.

- `src/prob_ml/sanity.py`
  Dataset QA command. Checks COCO bbox validity, category counts, missing image
  files, negative-only assumptions, split leakage, and writes bbox overlay
  images.

- `src/prob_ml/evaluate.py`
  Post-training evaluation command. Loads a checkpoint, evaluates validation and
  negative-only splits across multiple score thresholds, and writes
  false-positive / false-negative example visualizations.

- `src/prob_ml/yolo.py`
  Optional Ultralytics YOLO training entrypoint. This is not required for the
  main Faster R-CNN path.

- `src/prob_ml/cli.py`
  Adds CLI commands:
  `sanity-check`, `evaluate`, and `train-yolo`.

- `jobs/sanity-check.sbatch`
  DCC job for dataset QA.

- `jobs/evaluate.sbatch`
  DCC job for checkpoint evaluation.

- `jobs/train-yolo.sbatch`
  Optional DCC job for YOLO training.

- `tests/test_detector.py`
  Lightweight tests for detector matching and dataset indexing.

- `tests/test_sanity.py`
  Lightweight tests for COCO sanity-check behavior.

## Recommended Commands

Run dataset conversion after Owner 1 has produced render outputs:

```bash
uv run pest-pipeline convert --config configs/base.json
```

Check dataset quality before training:

```bash
uv run pest-pipeline sanity-check --config configs/base.json
```

Run a smoke training job on DCC:

```bash
uv run pest-pipeline dcc-submit --config configs/dcc_gpu_smoke.json --job train
```

Run full Faster R-CNN training on DCC:

```bash
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job train
```

Evaluate the trained checkpoint:

```bash
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job evaluate
```

Run single-image inference:

```bash
uv run pest-pipeline infer --config configs/base.json
```

Optional YOLO smoke/full training:

```bash
uv run pest-pipeline dcc-submit --config configs/dcc_gpu_smoke.json --job train-yolo
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job train-yolo
```

## Config Additions

The `training` section now supports:

- `pretrained`
  Whether to use pretrained torchvision detector weights. Defaults to `false`
  to avoid unexpected downloads.

- `augmentation`
  Lightweight training augmentation, currently horizontal flip and color jitter.

- `threshold_sweep`
  Confidence thresholds for TDR/FPR reporting, currently `[0.3, 0.5, 0.7]`.

New `sanity` section:

- `output_dir`
  Where dataset QA reports and overlay images are written.

- `max_overlay_images`
  Maximum overlay examples per split.

New `evaluation` section:

- `output_dir`
  Where detector evaluation reports and failure examples are written.

- `checkpoint`
  Detector checkpoint to evaluate.

- `thresholds`
  Confidence thresholds used for evaluation.

- `iou_threshold`
  IoU threshold used for same-class bbox matching.

- `max_failure_examples`
  Maximum false-positive / false-negative visualizations to write.

New `yolo` section:

- `data_yaml`
  Path to YOLO `data.yaml`.

- `model`
  Default is `yolov8n.pt`.

- `epochs`, `imgsz`, `batch`, `workers`, `device`, `name`
  Ultralytics training parameters.

## Output Artifacts

Expected main outputs:

- `artifacts/dataset/coco_train.json`
- `artifacts/dataset/coco_val.json`
- `artifacts/dataset/coco_neg_test.json`
- `artifacts/dataset/yolo/data.yaml`
- `artifacts/reports/dataset_sanity_report.json`
- `artifacts/reports/sanity_overlays/`
- `artifacts/models/detector/detector.pt`
- `artifacts/models/detector/training_report.json`
- `artifacts/reports/evaluation/detector_evaluation_report.json`
- `artifacts/reports/evaluation/failure_examples/`
- `artifacts/infer/predictions.json`
- `artifacts/infer/infer_result.jpg`

Generated artifacts should stay out of git.

## Important Design Notes

- Faster R-CNN is the main supported detector path because it uses existing
  torch/torchvision dependencies and directly consumes COCO annotations.
- YOLO is optional because it requires `ultralytics` and may download pretrained
  weights. The code path is available, but the main project should not depend
  on YOLO succeeding.
- There are no real pest-positive images. The strongest evaluation story is:
  synthetic or composited positive validation plus real kitchen negative-only
  false-positive testing.
- Split quality matters. Train/val/neg_test should be separated by background
  kitchen image, not just by generated frame.
- `sanity-check` should be run before training. It is the fastest way to catch
  bad boxes, category mistakes, missing files, and split leakage.
- `evaluate` should be run after training. It produces the threshold table and
  failure examples needed for the final presentation/report.

## Validation Already Run Locally

The following lightweight checks passed locally:

```bash
PYTHONPATH=src pytest -q tests/test_detector.py tests/test_sanity.py tests/test_dataset.py tests/test_video.py tests/test_manifest.py tests/test_cli.py tests/test_layout.py
python -m compileall src tests
PYTHONPATH=src python -m prob_ml.cli dcc-submit --config configs/dcc_gpu.json --job sanity-check
PYTHONPATH=src python -m prob_ml.cli dcc-submit --config configs/dcc_gpu.json --job evaluate
PYTHONPATH=src python -m prob_ml.cli dcc-submit --config configs/dcc_gpu.json --job train-yolo
```

Local machine does not currently have the full torch/ultralytics training
environment or final rendered dataset, so full training/evaluation was not run
locally.

## Suggested Next Steps

1. Owner 1 should produce or confirm rendered train/val outputs.
2. Owner 2 should run `convert` and `sanity-check`.
3. Fix any reported dataset errors before training.
4. Owner 3 should run smoke Faster R-CNN training.
5. If smoke training succeeds, run full Faster R-CNN training.
6. Run `evaluate` and collect threshold table plus failure examples.
7. Optionally run YOLO if `ultralytics` installs cleanly and time allows.
8. Update `SUBMISSION_DRAFT.md` with real metrics and selected visual examples.
