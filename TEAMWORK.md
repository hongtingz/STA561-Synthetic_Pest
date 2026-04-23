# Teamwork Plan

This document defines the split between the synthetic-data pipeline, dataset
packaging, and detector-training work.

## Current Project State

The project currently has:

- A `uv`-managed Python package under `src/prob_ml/`
- Config-driven local and DCC execution
- Single-image render smoke tests
- Manifest-driven batch render preparation
- A successful local batch render smoke output with frames and annotations
- DCC smoke configs and Slurm job scripts

The project still needs:

- Detector training
- Detector inference
- Evaluation for true detection rate and false positive rate
- Final interface cleanup between dataset export and detector training

## Owner 1 (Hongting): Synthetic Data And DCC Pipeline

Hongting is responsible for the data-generation and cluster-execution side.

Primary files:

- `src/prob_ml/render.py`
- `src/prob_ml/layout.py`
- `src/prob_ml/manifest.py`
- `src/prob_ml/blender/render_scene.py`
- `configs/*.json`
- `jobs/*.sbatch`
- `DCC_USAGE.md`

Responsibilities:

- Run and validate DCC render smoke tests.
- Scale `render-batch` from smoke tests to larger batches.
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

## Owner 2: Dataset Packaging And Evaluation Interface

This owner is responsible for turning generated render outputs into clean model
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
- Prepare ground-truth files needed by the detector owner.
- Coordinate with Owner 3 on any model-specific dataset loader needs.

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

Current Owner 2 implementation status:

- `src/prob_ml/dataset.py` now converts rendered frame annotations into COCO.
- The same conversion step also writes YOLO labels and `data.yaml`.
- File paths are written relative to the repository root where possible.
- Bounding boxes are validated against frame image size before export.
- The conversion step requires rendered `train` and `val` splits to exist.
- `neg_test` is treated as a real-image negative-only holdout, not a positive detection split.

## Owner 3: Detector Training, Inference, And Evaluation

This owner is responsible for the model side.

Primary files:

- `src/prob_ml/train.py`
- `src/prob_ml/infer.py`
- future model/helper modules under `src/prob_ml/`
- training-related config fields in `configs/*.json`

Responsibilities:

- Implement detector training in `src/prob_ml/train.py`.
- Use a detection model, not a pure image classifier.
- Read the packaged dataset produced by `src/prob_ml/dataset.py`.
- Save checkpoints and model artifacts under `artifacts/models/`.
- Implement inference in `src/prob_ml/infer.py`.
- Generate predicted bounding boxes, class labels, and confidence scores.
- Implement evaluation reports for detection quality.
- Track true detection rate and false positive rate.
- Coordinate with Owner 2 on COCO format and split names.

Expected outputs:

```text
artifacts/models/
  detector/
    checkpoints/
    final_model/

artifacts/eval/
  eval_report.json
  predictions.json
```

## Shared Dataset Contract

The handoff from Owner 1 to Owner 2 is batch-render output. The handoff from
Owner 2 to Owner 3 should be COCO-style detection data.

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
- Do not change category order without notifying all owners.

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
- Owner 3 should not need to know how Blender generated the frame.
- Owner 1 should not need to know model internals.
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

1. Owner 1 runs DCC smoke render:

```bash
uv run pest-pipeline dcc-submit --config configs/dcc_gpu_smoke.json --job render-batch
```

2. Owner 2 implements and tests dataset conversion:

```bash
uv run pest-pipeline convert --config configs/base.json
```

3. Owner 3 starts with a tiny local dataset from:

```text
artifacts/batch_render_smoke/kitchen_0001/
```

4. Owner 3 later switches to:

```text
artifacts/dataset/coco_train.json
artifacts/dataset/coco_val.json
artifacts/dataset/yolo/data.yaml
```

5. Owner 3 trains:

```bash
uv run pest-pipeline train --config configs/base.json
```

6. Owner 3 evaluates:

```bash
uv run pest-pipeline infer --config configs/base.json
```

## Model Guidance

The final model must localize and classify pests. A pure image classifier is not
enough because the project requires bounding boxes.

Reasonable detector options:

- Faster R-CNN with a lightweight backbone
- DETR-style detector
- YOLO-style detector if the dependency footprint stays manageable
- ViT-backed detector if time allows

For the first working baseline, prioritize a reliable detector pipeline over a
complex architecture.

## Evaluation Target

The instructor target is:

- At least 80% true detection rate
- Less than 5% false positive rate

The evaluation code should report:

- Overall true detection rate
- Overall false positive rate
- Per-class detection rates for mouse, rat, and cockroach
- Number of evaluated frames
- Number of ground-truth boxes
- Number of predicted boxes

Recommended evaluation split interpretation:

- `train` / `val`: rendered or composited positive pest data
- `neg_test`: real kitchen negative-only data for false-positive rate measurement
- Optional `test`: only if the team later creates a held-out positive rendered/composited split

## Coordination Rules

- Owner 1 owns render, layout, manifest, Blender, DCC, and render configs.
- Owner 2 owns dataset conversion, dataset validation, and split files.
- Owner 3 owns training, inference, model, and evaluation files.
- Shared config changes should be discussed before committing.
- Do not change the COCO dataset schema without notifying all owners.
- Do not change split names (`train`, `val`, `neg_test`, optional `test`) without notifying all owners.
- If Owner 3 trains with YOLO, the canonical class order must still match the COCO export.
- If a future positive `test` split is added, update both COCO and YOLO exports together.
- Keep generated artifacts out of git.
- Commit code and config changes, not rendered frames or model checkpoints.
