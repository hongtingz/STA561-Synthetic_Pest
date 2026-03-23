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

**Local demo first** — build a working Blender scene locally before scaling to cluster or adding the ViT training step.

Demo scope:
- Kitchen floor/room scene in Blender
- Animated pest (mouse) moving along a path
- Render frames + export bounding boxes
- Run via GUI (Blender Scripting tab) for easy iteration

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
