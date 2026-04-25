# Contributions

This document records the team contribution split across the synthetic-data
pipeline, dataset packaging, detector training, evaluation, and DCC
deployment work.

## Current Project State

The project currently has:

- A `uv`-managed Python package under `src/prob_ml/`
- Config-driven local and DCC execution
- Manifest-driven batch render preparation
- Successful local and DCC batch render outputs with frames and annotations
- DCC configs and Slurm job scripts for formal runs

The project still needs:

- Final detector training runs on rendered datasets
- Final inference examples and qualitative prediction figures
- Evaluation for true detection rate and false positive rate
- Final interface cleanup between dataset export, detector training, and reporting

## Hongting Zhang: Synthetic Data And DCC Pipeline

Hongting is responsible for the synthetic-data pipeline, DCC execution, and
cluster-side testing/debugging.

Primary files:

- `src/prob_ml/render.py`
- `src/prob_ml/layout.py`
- `src/prob_ml/manifest.py`
- `src/prob_ml/blender/render_scene.py`
- `configs/*.json`
- `jobs/*.sbatch`
- `DCC_DEPLOYMENT.md`

Responsibilities:

- Run and validate DCC render jobs.
- Test end-to-end pipeline stages on DCC and debug cluster-side failures.
- Scale `render-batch` to larger batches and maintain the formal DCC config.
- Keep Blender rendering, layout specs, and annotations stable.
- Keep DCC job scripts and config files documented and reproducible.
- Track render throughput, failure cases, and output quality.

Expected outputs:

```text
artifacts/batch_render*/
  kitchen_0001/
    frames/
    annotations.json
    layout.json
    layout_diagnostics.json
    layout_preview.svg
```

Submission-facing deliverables:

- Contribute the synthetic-data and DCC sections of the technical appendix.
- Document DCC testing, job submission, and cluster-side debugging decisions
  clearly enough for instructor reproduction.
- Document the end-to-end render workflow so the instructor can reproduce:
  kitchen photo -> layout spec -> rendered frames -> annotations.
- Provide figures or screenshots for the executive summary and FAQ showing
  representative generated kitchens, pests, and labeled outputs.
- If a notebook demo is included, own the render/demo portion that shows one
  image being converted into labeled synthetic frames.

## Russo Zhang: Dataset Packaging And Evaluation Interface

Russo is responsible for turning generated render outputs into clean model
inputs and evaluation-ready ground truth.

Primary files:

- `src/prob_ml/dataset.py`
- future dataset/helper modules under `src/prob_ml/`
- dataset-related config fields in `configs/*.json`
- dataset tests under `tests/`

Responsibilities:

- Implement dataset conversion in `src/prob_ml/dataset.py`.
- Read batch render outputs from `artifacts/batch_render*/`.
- Convert generated frame annotations into COCO-style detection datasets.
- Produce train/validation splits for positive rendered data.
- Preserve a separate `neg_test` split for real kitchen images with no pests.
- Validate that every annotation points to an existing frame image.
- Validate category IDs and bounding-box format.
- Export a YOLO-format dataset in addition to COCO for fast detector baselines.
- Prepare ground-truth files needed by the detector-training contributor.
- Coordinate with the detector-training contributor on any model-specific
  dataset loader needs.

Expected outputs:

```text
artifacts/dataset/
  coco_annotations.json
  coco_train.json
  coco_val.json
  coco_neg_test.json
  neg_test_images.json
  dataset_summary.json
  yolo/
    data.yaml
    images/
    labels/
```

Submission-facing deliverables:

- Contribute the dataset-packaging and annotation-format sections of the technical appendix.
- Document the COCO/YOLO export contract, split definitions, and negative-only
  holdout design clearly enough for full reproduction.
- Provide tables or summaries for the executive summary / FAQ describing dataset
  size, split counts, and category mapping.
- If a notebook demo is included, own the dataset-inspection portion that shows
  rendered annotations becoming training-ready files.

Current Russo implementation status:

- `src/prob_ml/dataset.py` now converts rendered frame annotations into COCO.
- The same conversion step also writes YOLO labels and `data.yaml`.
- File paths are written relative to the repository root where possible.
- Bounding boxes are validated against frame image size before export.
- If the manifest only contains rendered `unassigned` samples, the conversion
  step auto-assigns them into `train` and `val` for baseline training.
- `neg_test` is treated as a real-image negative-only holdout, not a positive detection split.

## Shuai Huang: Detector Training, Inference, And Evaluation

This contributor is responsible for the model side.

Primary files:

- `src/prob_ml/train.py`
- `src/prob_ml/infer.py`
- future model/helper modules under `src/prob_ml/`
- training-related config fields in `configs/*.json`

Responsibilities:

- Run and refine detector training in `src/prob_ml/train.py`.
- Use a detection model, not a pure image classifier.
- Read the packaged dataset produced by `src/prob_ml/dataset.py`.
- Save checkpoints and model artifacts under `artifacts/models/`.
- Run and refine inference in `src/prob_ml/infer.py`.
- Generate predicted bounding boxes, class labels, and confidence scores.
- Implement evaluation reports for detection quality.
- Track true detection rate and false positive rate.
- Coordinate with Russo on COCO format and split names.

Expected outputs:

```text
artifacts/models/
  detector/
    detector.pt
    training_report.json

artifacts/reports/evaluation/
  detector_evaluation_report.json
  failure_examples/

artifacts/infer/
  predictions.json
  infer_result.jpg
```

Submission-facing deliverables:

- Contribute the detector-training, inference, and evaluation sections of the
  technical appendix.
- Provide the main quantitative results for the executive summary:
  true detection rate, false positive rate, and key example predictions.
- Draft the FAQ answers about why the model was chosen, how it was evaluated,
  what failed, and what future improvements are most promising.
- If a notebook demo is included, own the training/evaluation demo showing how
  predictions are generated from the packaged dataset or held-out frames.

## Shared Dataset Contract

The handoff from Hongting to Russo is batch-render output. The handoff from
Russo to Shuai Huang should be COCO-style detection data.

Required categories:

```json
[
  {"id": 1, "name": "mouse"},
  {"id": 2, "name": "rat"},
  {"id": 3, "name": "cockroach"}
]
```

Category mapping contract:

- COCO uses `category_id` values `1=mouse`, `2=rat`, `3=cockroach`.
- YOLO uses zero-based class ids `0=mouse`, `1=rat`, `2=cockroach`.
- Do not change category order without notifying all contributors.

Expected COCO structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "artifacts/batch_render/kitchen_0001/frames/frame_00001.png",
      "width": 640,
      "height": 360
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 2,
      "bbox": [581.2, 236.9, 58.8, 32.5],
      "area": 1911.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "mouse"},
    {"id": 2, "name": "rat"},
    {"id": 3, "name": "cockroach"}
  ]
}
```

Notes:

- Bounding boxes use COCO format: `[x, y, width, height]`.
- Coordinates are pixel coordinates.
- `file_name` should point to the rendered frame image using a path relative to repo root.
- Shuai Huang should not need to know how Blender generated the frame.
- Hongting should not need to know model internals.
- `neg_test` contains real kitchen images with no pests and should be used for false-positive evaluation.
- Because the team does not currently have real positive pest images, any positive `test` split must come from rendered or composited data.

Expected YOLO structure:

```text
artifacts/dataset/yolo/
  data.yaml
  images/
    train/
    val/
    test/        # can point to neg_test if no positive test split exists
    neg_test/
  labels/
    train/
    val/
    test/
    neg_test/
```

## Recommended Parallel Workflow

1. Hongting runs formal DCC batch rendering:

```bash
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job render-batch
```

2. Russo implements and tests dataset conversion:

```bash
uv run pest-pipeline convert --config configs/dcc_gpu.json
```

3. Shuai Huang starts with a tiny local dataset from:

```text
artifacts/batch_render*/kitchen_0001/
```

4. Shuai Huang later switches to:

```text
artifacts/dataset/coco_train.json
artifacts/dataset/coco_val.json
artifacts/dataset/yolo/data.yaml
```

5. Russo checks dataset quality:

```bash
uv run pest-pipeline sanity-check --config configs/dcc_gpu.json
```

6. Shuai Huang trains:

```bash
uv run pest-pipeline train --config configs/dcc_gpu.json
```

7. Shuai Huang evaluates:

```bash
uv run pest-pipeline evaluate --config configs/dcc_gpu.json
```

8. Shuai Huang runs example inference:

```bash
uv run pest-pipeline infer --config configs/dcc_gpu.json
```

Optional YOLO comparison:

```bash
uv add ultralytics
uv run pest-pipeline train-yolo --config configs/dcc_gpu.json
```

## Model Guidance

The final model must localize and classify pests. A pure image classifier is not
enough because the project requires bounding boxes.

Reasonable detector options:

- Faster R-CNN with a lightweight backbone
- DETR-style detector
- YOLO-style detector if the dependency footprint stays manageable
- ViT-backed detector if time allows

The repo now includes a built-in Faster R-CNN path with augmentation, optional
pretrained weights, and threshold-sweep reporting. YOLO is available as an
optional comparison path through `pest-pipeline train-yolo`; keep Faster R-CNN
as the fallback if dependency installation or pretrained-weight downloads are
unreliable.

Use `pest-pipeline sanity-check` before model training to catch bbox, class, and
split-leakage problems. Use `pest-pipeline evaluate` after training to generate
the final threshold table and failure-case visualizations.

## Evaluation Target

The instructor target is:

- At least 80% true detection rate
- Less than 5% false positive rate

The evaluation code should report:

- Overall true detection rate
- Overall false positive rate
- Per-class detection rates for mouse, rat, and cockroach
- Number of evaluated frames
- Threshold-sweep values, especially around 0.3, 0.5, and 0.7, so the team can
  choose a confidence cutoff that controls false positives
- False-positive and false-negative example images for the final presentation
- Number of ground-truth boxes
- Number of predicted boxes

Recommended evaluation split interpretation:

- `train` / `val`: rendered or composited positive pest data
- `neg_test`: real kitchen negative-only data for false-positive rate measurement
- Optional `test`: only if the team later creates a held-out positive rendered/composited split

## Coordination Rules

- Hongting owns render, layout, manifest, Blender, DCC, and render configs.
- Russo owns dataset conversion, dataset validation, and split files.
- Shuai Huang owns training, inference, model, and evaluation files.
- Shared config changes should be discussed before committing.
- Do not change the COCO dataset schema without notifying all contributors.
- Do not change split names (`train`, `val`, `neg_test`, optional `test`) without
  notifying all contributors.
- If Shuai Huang trains with YOLO, the canonical class order must still match the COCO export.
- If a future positive `test` split is added, update both COCO and YOLO exports together.
- Keep generated artifacts out of git.
- Commit code and config changes, not rendered frames or model checkpoints.
