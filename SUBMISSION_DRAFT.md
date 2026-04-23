# Synthetic Data Generation for Pest Detection

Draft submission document for STA 561. This file is intended as a working
starting point and should be revised as the project, experiments, and final
results are completed.

## Executive Summary

### Problem

Pest detection is an important problem for commercial kitchen safety and
hygiene. In an ideal setting, a vision system could monitor a kitchen during
off-hours and identify mice, rats, or cockroaches before an infestation grows.
The main challenge is that high-quality labeled pest images in real kitchens are
difficult to obtain. Real pest events are relatively rare, manual bounding-box
annotation is time-consuming, and collecting a large balanced dataset across
multiple pest classes is impractical in the time available for this project.

### Our Approach

To address this data bottleneck, we built a synthetic-data pipeline centered on
Blender. The system takes a still kitchen image as input, extracts simple visual
cues from that image, builds an approximate kitchen layout, and renders labeled
synthetic frames containing pests. Because the pests and their trajectories are
generated programmatically, ground-truth bounding boxes are available
automatically. Those labeled frames are then packaged into standard computer
vision dataset formats for model training and evaluation.

Our goal is not to perform perfect single-image 3D reconstruction. Instead, we
aim to produce an adaptive approximation of the kitchen scene that preserves the
most important structure: camera viewpoint, rough room geometry, major fixtures,
and plausible pest motion near the floor and along kitchen surfaces. This
approximation makes the pipeline much more feasible while still supporting the
core objective of generating diverse labeled training data at scale.

### What We Built

The current system includes the following components:

- A config-driven Python package managed with `uv`
- A kitchen-photo-to-layout stage that estimates room and fixture structure from
  image cues
- A Blender rendering stage that supports three pest classes: mouse, rat, and
  cockroach
- Automatic frame-level bounding-box annotation generation
- Dataset export in both COCO and YOLO formats
- A negative-only real-kitchen holdout split for false-positive evaluation
- Duke Compute Cluster job scripts for larger rendering runs

This structure is designed to support an end-to-end workflow:

1. Input a kitchen photo
2. Generate a layout specification
3. Render synthetic labeled frames in Blender
4. Convert outputs into model-ready datasets
5. Train and evaluate a detector

### Why This Matters

The main contribution of this project is not just a model. It is a reproducible
data-generation and training pipeline. Synthetic rendering allows us to create
large amounts of labeled data without manual bounding-box annotation. This is
especially valuable for rare-event detection problems where positive examples
are difficult to collect in the real world.

The project also explores a practical middle ground between realism and
automation. Instead of requiring a perfect 3D kitchen model, we use a
photo-guided approximation and focus on building a pipeline that can scale to
many inputs and run on the Duke cluster.

### Current Status

At the time of this draft, the synthetic rendering and dataset-packaging stages
are implemented and tested. The repository can:

- Render pest-containing frames from kitchen-photo-derived layout specs
- Export frame annotations automatically
- Package those annotations into COCO and YOLO datasets
- Preserve real kitchen images with no pests as a separate negative-only split

Training and inference are still under active development. Final quantitative
results, including true detection rate and false positive rate, should be added
once detector experiments are complete.

### Preliminary Interpretation

Even before final detector experiments are complete, the project already
demonstrates that a structured synthetic-data pipeline for pest detection is
feasible. The system can generate labeled multi-class data with minimal manual
annotation effort and prepare it in a standard format for downstream model
training.

The largest remaining challenge is sim-to-real generalization. We currently have
real kitchen negatives for false-positive evaluation, but we do not yet have a
matched real positive pest dataset. This means our strongest claims should focus
on the pipeline, reproducibility, and training-readiness of the synthetic data,
while final generalization claims should be framed carefully.

### Future Work

If we had more time, the most important next steps would be:

- Train and compare one or more full object detectors on the packaged dataset
- Quantify false positives on real kitchen negatives more systematically
- Improve pest asset realism and scene diversity
- Expand the evaluation protocol with stronger held-out test settings
- Explore better compositing or domain-randomization strategies to reduce the
  sim-to-real gap

## FAQ

### 1. Why not train directly on real pest images?

Because that would require a large labeled dataset of mice, rats, and
cockroaches in real kitchens, including bounding boxes. That kind of dataset is
hard to collect, hard to annotate, and unlikely to be available at the scale
needed for detector training.

### 2. Why is synthetic data a good fit for this problem?

Synthetic data is especially useful when positive examples are rare and labels
are expensive. In a rendered scene, pest identity and location are known by
construction, so frame-level annotations can be produced automatically. This
makes it possible to generate a large amount of labeled training data at low
cost.

### 3. Does the system reconstruct a full 3D kitchen from one image?

No. The system does not attempt perfect single-image 3D reconstruction. Instead,
it extracts simple visual cues and uses them to generate a plausible kitchen
layout with the same overall perspective and structure. This is a deliberate
tradeoff between realism and practicality.

### 4. Why is an approximate layout acceptable?

For this project, the main objective is to generate useful labeled data for pest
detection, not to reproduce every physical detail of the kitchen. If the
rendered scene captures the viewpoint, major geometry, and plausible pest
placement well enough, it can still provide meaningful training data.

### 5. Why use Blender?

Blender is well suited for programmable scene construction, animation, and
rendering. It also provides a practical path to large-scale synthetic data
generation, especially when paired with Python automation and cluster
execution.

### 6. Why generate videos instead of only still images?

The project specification emphasizes video, but a video is ultimately a sequence
of frames. Treating the output as labeled frame sequences allows us to preserve
the spirit of the assignment while still using standard object-detection dataset
formats.

### 7. How are bounding boxes generated?

The renderer computes the projected image-space extents of each pest object
relative to the Blender camera. Those projected boxes are written automatically
for each rendered frame. This eliminates the need for manual annotation on the
synthetic data.

### 8. Why support three pest classes?

Because the assignment explicitly targets mice, rats, and cockroaches. A system
that only handled one class would not fully satisfy the project specification.

### 9. Why export both COCO and YOLO formats?

COCO is a widely used, standard format for object detection research and makes
the dataset contract explicit. YOLO format is convenient for fast detector
baselines and practical training workflows. Exporting both keeps the dataset
portable across model choices.

### 10. What is the role of the real kitchen images with no pests?

Those images are valuable as hard negatives. They help evaluate whether the
detector hallucinates pests in realistic kitchen backgrounds. This is important
because a model that performs well only on synthetic scenes may still generate
too many false positives on real images.

### 11. Why is there a `neg_test` split instead of a standard positive `test` split?

At present, we do not have real positive pest images. That means real-world
evaluation is strongest on the false-positive side. The `neg_test` split
preserves real no-pest kitchen images for that purpose. If a positive held-out
test split is added later, it will likely come from rendered or composited data.

### 12. What are the biggest limitations of the current system?

The main limitations are the realism gap between synthetic and real kitchens,
the lack of real positive pest images, and the fact that final detector training
and evaluation are still being completed. These limitations affect how strongly
we can claim real-world generalization.

### 13. What would improve the project most if more time were available?

The most impactful improvements would be stronger detector training, better pest
assets, more diverse backgrounds and camera conditions, and more realistic
evaluation on real positive examples.

### 14. Is the project reproducible?

Yes, that is one of the main design goals. The repository is organized around a
single Python package, JSON config files, Slurm job scripts, and standard output
directories. The technical appendix below explains the expected data flow and
commands.

## Technical Appendix

### 1. Project Goal

The goal of this project is to build an end-to-end pipeline that:

1. Takes a kitchen photo as input
2. Generates labeled synthetic pest frames or video
3. Exports frame-level bounding-box annotations
4. Packages the data for model training
5. Trains and evaluates a detector for mouse, rat, and cockroach localization

The assignment's A+ target additionally asks for an automated pipeline that can
run at scale on the Duke Compute Cluster and achieve strong detection quality on
held-out data.

### 2. System Overview

The current repository is organized as a pipeline with separate modules for:

- configuration loading
- kitchen-photo-driven layout generation
- Blender rendering
- annotation packaging
- cluster job submission
- training and inference entrypoints

The main intended data flow is:

`kitchen photo -> layout spec -> rendered frames -> annotations -> COCO/YOLO dataset -> detector training -> evaluation`

### 3. Repository Structure

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

### 4. Configuration And Execution

The project uses `uv` for environment and dependency management. This was chosen
to keep setup reproducible and lightweight.

Representative commands:

```bash
uv sync
uv run pest-pipeline doctor
uv run pest-pipeline plan --config configs/base.json
uv run pest-pipeline render-batch --config configs/local_render_batch_smoke.json
uv run pest-pipeline convert --config configs/base.json
uv run pest-pipeline dcc-submit --config configs/dcc_gpu_smoke.json --job render-batch
```

The JSON config files specify render settings, dataset paths, training paths,
and DCC resource requests.

### 5. Kitchen Photo To Layout

The layout stage is designed to be adaptive, not fully reconstructive. It
extracts simple cues from the input kitchen image, including brightness patterns
and structural signals, and uses those cues to define:

- room dimensions
- camera placement
- fixture layout
- pest motion paths

This produces a serialized layout specification that can be consumed by the
renderer.

### 6. Blender Rendering Pipeline

The rendering pipeline is implemented in Blender through Python. The renderer
currently supports:

- three pest classes: mouse, rat, and cockroach
- optional use of the source kitchen image as a photo background
- optional local 3D pest assets under `assets/pests/`
- batch rendering from a kitchen-photo manifest
- automatic per-frame annotation export

The current render configuration also supports CPU or GPU rendering depending on
local or cluster execution settings.

### 7. Annotation Generation

For each rendered frame, the Blender stage writes an annotation record that
contains:

- the frame index
- the rendered file path
- a list of visible pests
- an image-space bounding box for each visible pest

Bounding boxes use pixel coordinates and are later converted into standard
object-detection formats.

### 8. Dataset Packaging

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

### 9. Split Design

The current split design separates positive synthetic data from real negative
images:

- `train`
  Positive rendered or composited data used for model fitting
- `val`
  Positive rendered or composited data used for validation
- `neg_test`
  Real kitchen images with no pests, used to study false positives

This design reflects the current data reality: we have access to real kitchen
backgrounds without pests, but not a large real positive pest dataset.

### 10. Real Kitchen Negatives

A teammate identified a collection of real kitchen images that contain no pests.
These images are important for evaluation because they provide a more realistic
background distribution than synthetic scenes alone. In this project, they are
treated as a negative-only holdout to estimate how often a detector fires on
clean kitchen images.

### 11. Current Implementation Status

At the time of this draft:

- render preparation and Blender integration are implemented
- multi-class pest rendering support is implemented
- dataset conversion to COCO and YOLO is implemented
- DCC configuration and job scripts are present
- training is still a placeholder
- inference and evaluation are still placeholders

This means the project already contains a substantial amount of the data and
engineering pipeline, but the final model and metric-reporting stage are still
in progress.

### 12. Experimental Results

This section should be updated once detector experiments are complete.

Suggested subsections:

- Dataset size and split counts
- Detector architecture(s) used
- Training hyperparameters
- Quantitative metrics:
  - true detection rate
  - false positive rate
  - per-class performance
- Qualitative examples:
  - successful detections
  - false positives
  - failure cases

Placeholder note:

> Final detector results are not yet inserted into this draft.

### 13. Reproducibility Notes

To reproduce the current pipeline, a reader should have:

- the repository
- the `uv` environment
- Blender installed and accessible
- the kitchen image corpus referenced in the manifest
- optional local pest assets under `assets/pests/`

The technical workflow is intended to be reproducible from config files and
documented command-line entrypoints. The cluster execution path is documented in
`DCC_USAGE.md`.

### 14. Limitations

The most important current limitations are:

- no large real positive pest dataset
- a remaining realism gap between synthetic and real images
- dependence on asset quality for visual fidelity
- training and inference stages still under active development

These limitations matter because they affect how strongly we can generalize
results beyond the synthetic or semi-synthetic setting.

### 15. Future Work

The most promising next steps are:

- complete detector training on the exported datasets
- compare at least one practical detector baseline, such as YOLO
- improve qualitative realism of pest assets and scene interactions
- increase background diversity through real-image-driven generation
- refine evaluation on real negative images and future real positive data
- add richer result visualizations and notebook demos for the final submission

## Drafting Notes For Final Submission

Before final turn-in, this draft should be updated with:

- final quantitative metrics
- final detector choice and justification
- dataset size and split statistics
- selected rendered examples and prediction figures
- any notebook links or demo references
- polished wording for the executive summary once results are stable
