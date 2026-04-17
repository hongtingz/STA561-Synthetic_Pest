# README.md

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

**Local pipeline complete and evaluated.** ViT training passed targets (TDR=100%, FPR=0%). Next step: scale to DCC (add rat/cockroach pests there) and improve sim-to-real transfer.

Pipeline stages (all implemented):
1. `blender --background --python demo/render_demo.py` → renders 60 frames + raw annotations (`output/annotations.json`)
2. `python demo/to_coco.py` → converts to COCO format (`output/coco_annotations.json`)
3. `python train_vit.py` → fine-tunes ViT, evaluates TDR/FPR, saves model to `output/vit_model/`
4. `python infer.py path/to/image.jpg` → runs multi-scale sliding window inference, saves annotated result to `output/infer_result.jpg`

Optional: `python demo/visualize.py` → draws bboxes on frames, saves to `output/viz/`

## Rendering

Local demo uses **Cycles renderer** (64 samples + denoising, Filmic tone mapping) for realistic output. The scene includes:
- Procedural materials: checker tile floor, painted drywall walls, wood-grain cabinets, granite countertop, fur-textured mouse
- Kitchen geometry: lower/upper cabinets, countertop, fridge, side table with legs, baseboards
- 4-light setup: warm overhead key, under-cabinet strip, cool side fill (window sim), rim light
- Multi-part mouse model: body, head, snout, nose, ears, eyes, bezier tail
- Local demo is **mouse-only**; rat and cockroach models will be added on DCC

## Training Results

- TDR: 100%, FPR: 0% — **PASS** (targets: ≥80% TDR, <5% FPR)
- Evaluated on synthetic val split (80/20 split of 60 rendered frames)
- Note: metrics are on synthetic data only; sim-to-real transfer not yet validated

## Known Issues / Fixes Applied

- Mouse animation path was `x=-3→x=3`, causing frames 1–35 to have off-screen bboxes (negative widths). Fixed to `x=-1.5→x=1.5` so mouse stays within camera frustum all 60 frames.
- `visualize.py` and `to_coco.py` both guard against invalid bboxes (`width <= 0 or height <= 0`).
- Pylance warnings on `bpy` node socket attributes (e.g. `default_value`, `color_ramp`) are false positives — `bpy` stubs are incomplete but the code runs correctly in Blender.

## Environment

- Managed with [`uv`](https://docs.astral.sh/uv/) — project venv at `.venv` (Python 3.12)
- Dependencies declared in `pyproject.toml`, locked in `uv.lock`
- Sync deps: `uv sync`
- Run scripts: `uv run python train_vit.py` (or activate with `source .venv/bin/activate`)
- `bpy`/`mathutils` are Blender-bundled — not installed via `uv`, only available inside Blender's Python (use `blender --background --python demo/render_demo.py`)

## Key Design Decisions

- **Blender 5.0.1** installed at `/Applications/Blender.app`, CLI available as `blender`
- For demo: use Blender GUI scripting tab; for production: headless via `blender --background --python script.py`
- Annotations are generated programmatically from known 3D ground truth (no human labeling)
- Target COCO-format JSON for compatibility with standard detection frameworks
- ViT fine-tuning uses crop-based binary classification (pest/no-pest) — not full object detection — because 60 frames is too little data for a full detector
- Negative samples: 3 random background crops per frame that don't overlap any annotated bbox
- Training: AdamW lr=2e-5, batch 8, 10 epochs, 80/20 train/val split
- Inference (`infer.py`): multi-scale sliding window (60/100/150/200px), each crop resized to 224×224, NMS applied — needed because mouse bbox (~61×42px) is much smaller than 224×224
