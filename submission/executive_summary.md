# Executive Summary

## Problem

Pest detection matters for commercial kitchen hygiene and safety. In a practical
deployment, a vision system could monitor kitchens during off-hours and detect
mice, rats, or cockroaches before an infestation becomes severe. The main data
challenge is that high-quality labeled pest images in real kitchens are hard to
obtain. Real pest events are rare, manual bounding-box annotation is expensive,
and collecting a balanced dataset across multiple pest classes is difficult
within the scope of a course project.

## Approach

To address this bottleneck, we built a synthetic-data pipeline centered on
Blender and Python automation. The system takes a still kitchen image as input,
extracts simple visual cues from that image, builds an approximate kitchen
layout, and renders labeled synthetic pest frames. Because the pests and their
motion paths are generated programmatically, bounding boxes are available
automatically without manual annotation. The rendered outputs are then packaged
into standard object-detection formats for downstream model training and
evaluation.

Our goal is not perfect single-image 3D reconstruction. Instead, we aim for an
adaptive scene approximation that preserves the most important structure for the
task: camera viewpoint, room geometry, large fixtures, and plausible pest
motion near the floor and along kitchen surfaces. This tradeoff makes the
pipeline more feasible while still supporting large-scale labeled data
generation.

## What We Built

The current system includes:

- a Python workflow for running the pipeline consistently
- a kitchen-photo-to-layout stage that estimates room and fixture structure
  from image cues
- a Blender rendering stage that supports mouse, rat, and cockroach classes
- automatic frame-level bounding-box annotation generation
- export into standard detector-training dataset formats
- a negative-only real-kitchen holdout split for false-positive evaluation
- Duke Compute Cluster job scripts for larger rendering and training runs
- a lightweight notebook demo for reviewing generated outputs

The current end-to-end workflow is:

1. Input a kitchen photo.
2. Generate a layout specification.
3. Render synthetic labeled frames or video in Blender.
4. Convert outputs into model-ready datasets.
5. Train and evaluate a detector.

## Results

The synthetic rendering, dataset conversion, sanity checking, detector
training, checkpoint evaluation, and DCC deployment scaffolding are all in
place. The repository can:

- render pest-containing frames from kitchen-photo-derived layout specs
- export frame annotations automatically
- package those annotations into COCO and YOLO datasets
- preserve real kitchen images with no pests as a separate negative-only split
- train and evaluate a transformer-style detector

Model training and inference now use a ViT-style object detector by default:
the config value `vit` maps to the Hugging Face `hustvl/yolos-tiny` detector.
The same pipeline also keeps Faster R-CNN and YOLO-compatible exports available
for comparison. This lets us evaluate detector choices without changing the
synthetic-data generation or annotation pipeline.

The most important current conclusion is that the synthetic-data pipeline is
operational and reproducible. It already demonstrates that kitchen-photo-driven
rendering, automatic annotation generation, dataset packaging, and DCC-based
execution can be integrated into a single workflow.

## Future Work

The biggest remaining challenge is sim-to-real generalization. We currently
have real kitchen negatives for false-positive evaluation, but we do not yet
have a matched real positive pest dataset. As a result, the strongest current
claims should focus on the pipeline itself, reproducibility, and
training-readiness of the synthetic data rather than final real-world
performance.

The most important active result updates are:

- complete the DCC render and ViT/YOLOS-tiny training runs
- improve pest asset realism, placement, and scene diversity
- quantify false positives on real kitchen negatives more systematically
- refresh quantitative tables and qualitative examples from the generated
  training/evaluation reports
