# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

STA 561 Final Project — **Synthetic Data Generation for Pest Detection** (Problem 2 from `final_project_2026-1.pdf`).

The goal is an end-to-end Python pipeline that:
1. Takes a kitchen photo as input
2. Generates labeled synthetic video (30–60s) of pests (mice, rats, cockroaches) in that kitchen using Blender's Python API (`bpy`)
3. Exports per-frame bounding box annotations (COCO format)
4. Trains a Vision Transformer (ViT) on the synthetic data
5. Must achieve ≥80% true detection rate and <5% false positive rate

A+ requirement: fully automated pipeline runnable at scale on the Duke compute cluster.

## Current Focus

**Local pipeline complete** — full end-to-end pipeline is working locally. Next step: run ViT training, evaluate metrics, then scale to DCC if needed.

Pipeline stages (all implemented):
1. `blender --background --python demo/render_demo.py` → renders 60 frames + raw annotations (`output/annotations.json`)
2. `python demo/to_coco.py` → converts to COCO format (`output/coco_annotations.json`)
3. `python train_vit.py` → fine-tunes ViT, evaluates TDR/FPR, saves model to `output/vit_model/`

Optional: `python demo/visualize.py` → draws bboxes on frames, saves to `output/viz/`

## Known Issues / Fixes Applied

- Mouse animation path was `x=-3→x=3`, causing frames 1–35 to have off-screen bboxes (negative widths). Fixed to `x=-1.5→x=1.5` so mouse stays within camera frustum all 60 frames.
- `visualize.py` and `to_coco.py` both guard against invalid bboxes (`width <= 0 or height <= 0`).

## Environment

- Conda env: `pest_ml` (Python 3.11) at `/opt/homebrew/Caskroom/miniconda/base/envs/pest_ml`
- Activate: `conda activate pest_ml`
- Install deps: `pip install -r requirements.txt`
- `bpy`/`mathutils` are Blender-bundled — not installed in `pest_ml`, only available inside Blender's Python

## Key Design Decisions

- **Blender 5.0.1** installed at `/Applications/Blender.app`, CLI available as `blender`
- For demo: use Blender GUI scripting tab; for production: headless via `blender --background --python script.py`
- Annotations are generated programmatically from known 3D ground truth (no human labeling)
- Target COCO-format JSON for compatibility with standard detection frameworks
- ViT fine-tuning uses crop-based binary classification (pest/no-pest) — not full object detection — because 60 frames is too little data for a full detector
- Negative samples: 3 random background crops per frame that don't overlap any annotated bbox
- Training: AdamW lr=2e-5, batch 8, 10 epochs, 80/20 train/val split
