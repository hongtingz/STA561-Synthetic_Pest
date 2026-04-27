# Technical Appendix

## 1. Project Goal

The goal of this project is to build an end-to-end pipeline that:

1. Takes a kitchen photo as input.
2. Generates labeled synthetic pest frames or video.
3. Exports frame-level bounding-box annotations.
4. Packages the data for model training.
5. Trains and evaluates a detector for mouse, rat, and cockroach localization.

The assignment's A+ target additionally asks for an automated pipeline that can
run at scale on the Duke Compute Cluster and achieve strong detection quality
on held-out data.

## 2. System Overview

The current repository is organized as a pipeline with separate modules for:

- configuration loading
- kitchen-photo-driven layout generation
- Blender rendering
- annotation packaging
- cluster job submission
- training and inference entrypoints
- ViT/YOLOS-tiny, Faster R-CNN, and optional YOLO detector paths that reuse the
  same exported dataset and evaluation contract

The main intended data flow is:

`kitchen photo -> layout spec -> rendered frames -> annotations -> COCO/YOLO dataset -> detector training -> evaluation`

The main implemented detector path is:

`kitchen photo -> layout spec -> rendered frames -> annotations -> COCO export -> sanity-check -> ViT/transformer detector training -> threshold selection -> evaluation on validation and neg_test`

## 3. Repository Structure

Important directories include:

- `src/prob_ml/`
  Main Python package
- `configs/`
  Runtime configuration files
- `jobs/`
  Slurm job scripts for Duke cluster usage
- `assets/pests/`
  Optional local 3D pest assets and attribution notes
- `tests/`
  Lightweight validation tests
- `artifacts/`
  Generated outputs such as rendered frames, packaged datasets, and model files
- `notebooks/`
  DCC-oriented notebook demos for instructor-facing review

## 3a. Core modules (implementation map)

The CLI entrypoint is `pest-pipeline`, defined in `pyproject.toml` as
`prob_ml.cli:main`. The following modules are the main implementation anchors:

- `prob_ml/cli.py`: command-line entrypoint for `plan`, `render`,
  `render-batch`, `convert`, `sanity-check`, `train`, `evaluate`,
  `train-yolo`, `infer`, `pipeline`, `doctor`, and `dcc-submit`.
- `prob_ml/config.py`: loads JSON into `PipelineConfig` and resolves paths
  relative to the repo root.
- `prob_ml/manifest.py`: parses the kitchen-photo CSV manifest into
  `KitchenPhotoRecord` rows.
- `prob_ml/layout.py`: extracts image cues and writes `layout.json`.
- `prob_ml/render.py`: prepares batch renders and invokes Blender.
- `prob_ml/blender/render_scene.py`: Blender-side scene construction,
  animation, bbox projection, and frame export.
- `prob_ml/video.py`: optional H.264 mux of numbered PNG frames via `ffmpeg`.
- `prob_ml/dataset.py`: merges batch renders and manifest splits into COCO and
  YOLO exports.
- `prob_ml/sanity.py`: dataset integrity checks and bbox overlay images.
- `prob_ml/detector.py`: ViT/YOLOS-tiny and Faster R-CNN model builders,
  `CocoDetectionDataset`, IoU matching, and TDR/FPR summaries.
- `prob_ml/train.py`: training loop, checkpoints, `training_report.json`, and
  threshold sweep on validation and `neg_test`.
- `prob_ml/evaluate.py`: checkpoint evaluation and failure-case images.
- `prob_ml/infer.py`: single-image prediction JSON and visualization.
- `prob_ml/yolo.py`: optional Ultralytics YOLO training.
- `prob_ml/dcc.py`: builds `sbatch` commands from config and `jobs/*.sbatch`.
- `prob_ml/pipeline.py`: creates artifact directories and prints a resolved
  plan summary.

Automated checks live under `tests/` (for example `test_cli.py`, `test_dataset.py`,
`test_manifest.py`, `test_detector.py`).

## 3b. Kitchen photo manifest (CSV)

Batch rendering expects a CSV at the path given by `inputs.kitchen_manifest`
(default `data/raw/kitchen/metadata/manifest.csv`). Each row describes one
kitchen image. The loader (`prob_ml/manifest.py`) expects:

- **`relative_path`** or **`filename`**: path to the image file (relative to the
  repo root or absolute). At least one must be set.
- **`image_id`**: optional stable id; defaults to the file stem if omitted.
- **`split`**: `train`, `val`, `test`, `neg_test`, `unassigned`, etc. Rows with
  `unassigned` can be auto-partitioned into `train`/`val` during `convert` when
  render outputs exist (see `dataset.py`).
- **`enabled`**: optional; `false`/`0`/`no` skips the row when
  `render.batch_enabled_only` is true.

Every referenced image file must exist at load time. The repository does not
ship a full kitchen corpus; reviewers should place images under
`data/raw/kitchen/images/` (or paths listed in the manifest) per
`README.md` / `DCC_DEPLOYMENT.md`.

## 3c. Python runtime and dependencies

`pyproject.toml` declares:

- **Python**: `>=3.12`
- **Runtime packages**: `pillow`, `torch`, `torchvision`, `transformers`

The main detector path uses **Hugging Face Transformers** through
`hustvl/yolos-tiny`. The config aliases `vit`, `vit_detector`, `yolos`,
`yolos-tiny`, and `yolos_tiny` all resolve to that model id. The same
`pest-pipeline train`, `evaluate`, and `infer` entrypoints also support
`fasterrcnn_mobilenet_v3_large_320_fpn` as a torchvision fallback.

**YOLO**: `pest-pipeline train-yolo` requires **`ultralytics`**, which is
**not** pinned in `pyproject.toml`. Install with `uv add ultralytics` (or
equivalent) if that baseline is needed.

## 3d. Default training, evaluation, and YOLO settings (`configs/dcc_gpu.json`)

These values are authoritative for the main DCC profile (edit the JSON to change
behavior):

**Training (ViT/YOLOS-tiny)**

- Model alias: `vit`, resolved to `hustvl/yolos-tiny`
- Epochs: `12`, batch size: `4`, optimizer: **AdamW**, learning rate: `2e-5`,
  weight decay: `1e-4`
- `pretrained`: `true` by default for the main DCC config
- `transformer_image_size`: `640`
- `checkpoint_interval`: `3`; checkpoint resume is supported
- Augmentation: horizontal flip probability `0.5`; color jitter brightness
  `0.2`, contrast `0.2`, saturation `0.1`
- Matching: training `score_threshold` `0.3`, IoU threshold `0.5`
- Threshold sweep list: `[0.3, 0.5, 0.7]` (reported for both val and `neg_test`
  in `training_report.json`)

**Evaluation**

- Checkpoint default: `artifacts/models/detector/detector.pt`
- Report directory: `artifacts/reports/evaluation/`
- Thresholds: `[0.3, 0.5, 0.7]`, IoU `0.5`, up to `12` failure-example images

**YOLO (optional)**

- `yolov8n.pt`, epochs `20`, `imgsz` `640`, batch `8`, workers `4`

**Batch render (DCC)**

- `batch_limit`: `25`, `batch_enabled_only`: `true`, `scene_seed`: `42`
- `pest_asset_style`: `hybrid`, `photo_background`: `true`

## 3e. Metrics reported in code

`prob_ml/detector.py` defines project-level summaries used in training and
evaluation:

- **True detection rate (TDR)**: `matched_boxes / ground_truth_boxes` over the
  evaluated split, using greedy same-class matching and a configurable IoU
  threshold.
- **False positive rate (FPR)** (image-level): `false_positive_images /
  evaluated_images`. In `match_prediction_to_target`, `false_positive_images` is
  `1` only when the image has **no** ground-truth boxes but **at least one**
  prediction clears the score threshold. That makes the metric align with
  **negative-only** evaluation (`neg_test`). On splits that contain positives,
  extra unmatched predictions do **not** increment this image-level counter
  (they still affect TDR via unmatched ground truth).

Per-class detection rates are also aggregated for qualitative diagnosis.

## 3f. Inference behavior

`pest-pipeline infer` runs the detector on **one** RGB image with a **single**
full-frame forward pass (after `to_tensor` and device placement). The config
keys `inference.window_sizes` and `inference.stride_ratio` appear in JSON configs
but are **not** read by the current `infer.py` implementation; they are
reserved for a possible future multi-scale or tiled inference path.

## 4. Configuration And Execution

The project uses `uv` for environment and dependency management. This was
chosen to keep setup reproducible and lightweight.

Representative commands:

```bash
uv sync
uv run pest-pipeline doctor --config configs/dcc_gpu.json
uv run pest-pipeline plan --config configs/dcc_gpu.json
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job render-batch
uv run pest-pipeline convert --config configs/dcc_gpu.json
uv run pest-pipeline dcc-submit --config configs/dcc_gpu.json --job train
uv run pest-pipeline evaluate --config configs/dcc_gpu.json
```

The JSON config files specify render settings, dataset paths, training paths,
evaluation paths, and DCC resource requests.

## 5. Kitchen Photo To Layout

The layout stage is designed to be adaptive rather than fully reconstructive.
It extracts simple cues from the input kitchen image, including brightness
patterns and structural signals, and uses those cues to define:

- room dimensions
- camera placement
- fixture layout
- pest motion paths

This produces a serialized layout specification that can be consumed by the
renderer.

## 6. Blender Rendering Pipeline

The rendering pipeline is implemented in Blender through Python. The renderer
currently supports:

- three pest classes: mouse, rat, and cockroach
- optional use of the source kitchen image as a photo background
- optional local 3D pest assets under `assets/pests/`
- batch rendering from a kitchen-photo manifest
- automatic per-frame annotation export
- GPU-oriented cluster execution through Slurm jobs and JSON configuration

The current formal DCC render configuration in `configs/dcc_gpu.json` uses:

- `fps = 30`
- `seconds = 30`
- `resolution = 1280 x 720`
- `samples = 64`
- `render_device = GPU`
- `compute_backend = CUDA`

## 7. Annotation Generation

For each rendered frame, the Blender stage writes an annotation record that
contains:

- the frame index
- the rendered file path
- a list of visible pests
- an image-space bounding box for each visible pest

Bounding boxes use pixel coordinates and are later converted into standard
object-detection formats.

## 8. Dataset Packaging

The dataset-packaging stage reads per-background render outputs from
`artifacts/batch_render*/` and exports:

- COCO training data
- COCO validation data
- a combined COCO positive dataset
- a COCO-style negative-only holdout
- a YOLO dataset with `images/`, `labels/`, and `data.yaml`
- a summary file with split counts and category mappings

Current category mapping:

- COCO: `1=mouse`, `2=rat`, `3=cockroach`
- YOLO: `0=mouse`, `1=rat`, `2=cockroach`

The packaging step validates:

- that annotated frame files exist
- that pest labels are recognized
- that bounding boxes are positive and within image bounds

If the manifest contains rendered scenes marked only as `unassigned`, the
converter can auto-assign those rendered scenes into `train` and `val` for the
baseline training flow while preserving real kitchen images as `neg_test`.

## 9. Split Design

The current split design separates positive synthetic data from real negative
images:

- `train`
  Positive rendered or composited data used for model fitting
- `val`
  Positive rendered or composited data used for validation
- `neg_test`
  Real kitchen images with no pests, used to study false positives

This design reflects the current data reality: the team has access to real
kitchen backgrounds without pests, but not a large real positive pest dataset.

## 10. Real Kitchen Negatives

The project includes a collection of real kitchen images that contain no pests.
These images are important for evaluation because they provide a more realistic
background distribution than synthetic scenes alone. In the current pipeline,
they are treated as a negative-only holdout used to estimate how often a
detector fires on clean kitchen images.

## 11. DCC Deployment

The DCC workflow is documented in `DCC_DEPLOYMENT.md`. The main formal cluster
sequence is:

```text
render-batch -> convert -> sanity-check -> train -> evaluate -> infer
```

Important DCC-facing components:

- `configs/dcc_gpu.json` (primary), `configs/base.json` (local CPU-oriented),
  `configs/dcc_gpu_smoke.json` (short validation)
- `jobs/render.sbatch`, `jobs/render-batch.sbatch`, `jobs/pipeline.sbatch`
- `jobs/sanity-check.sbatch`, `jobs/train.sbatch`, `jobs/evaluate.sbatch`,
  `jobs/train-yolo.sbatch`
- `scripts/dcc_submit.sh`
- `notebooks/dcc_pipeline_demo.ipynb`

`pest-pipeline dcc-submit --job <name>` accepts: `pipeline`, `render`,
`render-batch`, `sanity-check`, `train`, `evaluate`, `train-yolo`.

The notebook is intentionally read-only with respect to heavy pipeline stages.
It is designed for instructor-facing artifact review after the jobs have
already completed.

## 12. Current Implementation Status

At the current stage:

- render preparation and Blender integration are implemented
- multi-class pest rendering support is implemented
- dataset conversion to COCO and YOLO is implemented
- dataset sanity checking and bounding-box overlay generation are implemented
- DCC configuration and job scripts are present
- a ViT/YOLOS-tiny detector training path is implemented and selected by the
  main DCC config
- Faster R-CNN remains implemented as a lightweight fallback
- detector training supports augmentation, optional pretrained weights,
  checkpoint resume, epoch checkpoints, and threshold-sweep reporting
- checkpoint evaluation supports validation and negative-holdout reports plus
  failure-case visualizations
- an optional YOLO training entrypoint is implemented
- single-image detector inference is implemented
- optional **ffmpeg** mux of rendered frames to MP4 when `render.mux_video` is
  true (`prob_ml/video.py`; default is **true** if the key is omitted, so `ffmpeg`
  may be needed unless muxing is disabled in JSON)
- longer DCC render/train/evaluate jobs can update the report artifacts without
  changing the code path

This means the repository already contains a substantial amount of the data and
engineering pipeline, with model-selection and metric reporting driven by
generated DCC artifacts.

## 13. Experimental results and report artifacts

**Where numbers land after training**

The codebase writes structured reports at stable paths so later DCC runs can
refresh the quantitative tables and qualitative figures:

- Dataset summary: `artifacts/dataset/dataset_summary.json`.
  Contains split counts, category ids, and missing-background diagnostics.

- Sanity report: `artifacts/reports/dataset_sanity_report.json`.
  Contains integrity checks before training, including bbox and split checks.

- Training report: `artifacts/models/detector/training_report.json`.
  Contains per-epoch loss, validation TDR, `neg_test` metrics, saved checkpoint
  paths, and threshold sweep values.

- Evaluation report:
  `artifacts/reports/evaluation/detector_evaluation_report.json`.
  Contains multi-threshold evaluation summaries and failure-case paths.

**What to paste into the final write-up**

- Dataset size and split counts (from `dataset_summary.json`)
- Detector architecture: ViT/YOLOS-tiny through Hugging Face Transformers,
  plus Faster R-CNN or YOLOv8n if comparison runs are used
- Hyperparameters: see **Section 3d** and the committed `configs/dcc_gpu.json`
- Quantitative: TDR, image-level FPR on `neg_test`, per-class rates — from
  training/evaluation JSON
- Qualitative: overlays under `artifacts/reports/sanity_overlays/`, failure
  examples under `artifacts/reports/evaluation/failure_examples/`, notebook
  `notebooks/dcc_pipeline_demo.ipynb`

The repository is intentionally organized so the latest DCC artifacts can be
reviewed directly in the notebook and copied into final result tables without
rewriting the pipeline.

## 14. Reproducibility Notes

To reproduce the current pipeline, a reader should have:

- the repository
- the `uv` environment (`uv sync` installs `pyproject.toml` dependencies)
- Blender installed and on `PATH` (or path set in config as `blender_executable`)
- **`ffmpeg`** if video muxing is enabled in config
- the kitchen image corpus referenced by the manifest (default layout under
  `data/raw/kitchen/` per `configs/dcc_gpu.json` keys `kitchen_photo_dir` and
  `kitchen_manifest`)
- optional local pest assets under `assets/pests/` (see `assets/pests/CREDITS.md`
  for third-party model sources; large binaries may be omitted from git)
- access to the DCC account/partition configuration described in
  `DCC_DEPLOYMENT.md` when running on Duke

The technical workflow is intended to be reproducible from config files and
documented command-line entrypoints. The cluster execution path is documented in
`DCC_DEPLOYMENT.md` (including **Section 8** expected output paths), and the
generated outputs can be reviewed in `notebooks/dcc_pipeline_demo.ipynb`.

**Raw inputs and asset links:** the pipeline does not require real positive
pest imagery. Its required raw visual input is a kitchen-photo corpus listed in
`data/raw/kitchen/metadata/manifest.csv`; a reviewer can use the same
team-provided kitchen images or substitute another kitchen-photo folder with
the same manifest format. Optional pest assets and their raw source URLs are
documented in `assets/pests/CREDITS.md`. If those optional assets are
unavailable, the renderer uses procedural fallback geometry so that the code
path remains executable.

## 15. Limitations

The most important current limitations are:

- no large real positive pest dataset
- a remaining realism gap between synthetic and real images
- dependence on asset quality for visual fidelity
- final visual quality and detector comparisons depend on the latest DCC
  render/train/evaluate artifacts

These limitations matter because they affect how strongly the team can
generalize results beyond the synthetic or semi-synthetic setting.

## 16. Future Work

The most promising next steps are:

- use the latest DCC reports to refresh detector metric tables
- run the optional YOLO comparison if `ultralytics` and pretrained weights are
  available in the training environment
- improve qualitative realism of pest assets and scene interactions
- increase background diversity through real-image-driven generation
- refine evaluation on real negative images and future real positive data
- compare ViT/YOLOS-tiny against Faster R-CNN and YOLO on the same COCO export
- expand the existing DCC notebook demo with richer result visualizations for
  the final submission
