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
- a planned second detector path that reuses the same exported dataset and
  evaluation contract

The main intended data flow is:

`kitchen photo -> layout spec -> rendered frames -> annotations -> COCO/YOLO dataset -> detector training -> evaluation`

The current implemented detector path is:

`kitchen photo -> layout spec -> rendered frames -> annotations -> COCO export -> Faster R-CNN baseline -> evaluation`

The planned A+-oriented extension is:

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

- `configs/dcc_gpu.json`
- `jobs/render-batch.sbatch`
- `jobs/sanity-check.sbatch`
- `jobs/train.sbatch`
- `jobs/evaluate.sbatch`
- `scripts/dcc_submit.sh`
- `notebooks/dcc_pipeline_demo.ipynb`

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
- a lightweight Faster R-CNN detector training baseline is implemented
- detector training supports augmentation, optional pretrained weights, and
  threshold-sweep reporting
- checkpoint evaluation supports validation and negative-holdout reports plus
  failure-case visualizations
- an optional YOLO training entrypoint is implemented
- single-image detector inference is implemented
- a transformer-detector upgrade path is planned on top of the same COCO export
  and DCC workflow
- final trained checkpoints and metric tables are still pending

This means the repository already contains a substantial amount of the data and
engineering pipeline, but the final model-selection and metric-reporting stage
are still in progress.

## 13. Experimental Results Placeholder

This section should be updated once detector experiments are complete.

Suggested subsections:

- dataset size and split counts
- detector architecture or architectures used
- training hyperparameters
- quantitative metrics:
  - true detection rate
  - false positive rate
  - per-class performance
- qualitative examples:
  - successful detections
  - false positives
  - failure cases

Placeholder note:

> Final detector results are not yet inserted into this appendix draft.

## 14. Reproducibility Notes

To reproduce the current pipeline, a reader should have:

- the repository
- the `uv` environment
- Blender installed and accessible
- the kitchen image corpus referenced in the manifest
- optional local pest assets under `assets/pests/`
- access to the DCC account/partition configuration described in
  `DCC_DEPLOYMENT.md`

The technical workflow is intended to be reproducible from config files and
documented command-line entrypoints. The cluster execution path is documented in
`DCC_DEPLOYMENT.md`, and the generated outputs can be reviewed in
`notebooks/dcc_pipeline_demo.ipynb`.

## 15. Limitations

The most important current limitations are:

- no large real positive pest dataset
- a remaining realism gap between synthetic and real images
- dependence on asset quality for visual fidelity
- final training runs and inference result selection still pending

These limitations matter because they affect how strongly the team can
generalize results beyond the synthetic or semi-synthetic setting.

## 16. Future Work

The most promising next steps are:

- complete detector training on the exported datasets
- run the optional YOLO comparison if `ultralytics` and pretrained weights are
  available in the training environment
- improve qualitative realism of pest assets and scene interactions
- increase background diversity through real-image-driven generation
- refine evaluation on real negative images and future real positive data
- add a ViT- or DETR-style detector on top of the same COCO export and compare
  it against the Faster R-CNN baseline
- expand the existing DCC notebook demo with richer result visualizations for
  the final submission
