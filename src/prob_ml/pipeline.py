"""Pipeline planning and orchestration helpers."""

from __future__ import annotations

from pathlib import Path

from prob_ml.config import PipelineConfig


def ensure_runtime_directories(config: PipelineConfig) -> list[Path]:
    """Create the main artifact and log directories from config."""
    repo_root = config.repo_root
    created: list[Path] = []

    candidate_paths = [
        "artifacts",
        "data/inputs",
        config.section("inputs").get("kitchen_photo_dir"),
        config.section("inputs").get("kitchen_manifest"),
        "data/intermediate/layout",
        "logs/dcc",
        config.section("inputs").get("layout_diagnostics"),
        config.section("inputs").get("layout_preview"),
        config.section("render").get("batch_output_dir"),
        config.section("dataset").get("frames_dir"),
        config.section("dataset").get("annotations_raw"),
        config.section("dataset").get("video_output"),
        config.section("dataset").get("coco_annotations"),
        config.section("training").get("output_dir"),
        config.section("inference").get("output_image"),
    ]

    for item in candidate_paths:
        if not item:
            continue
        path = repo_root / item
        target_dir = path if path.suffix == "" else path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        if target_dir not in created:
            created.append(target_dir)

    return created


def render_plan(config: PipelineConfig) -> str:
    """Summarize the resolved pipeline layout."""
    render = config.section("render")
    dataset = config.section("dataset")
    training = config.section("training")
    inference = config.section("inference")
    dcc = config.section("dcc")
    pipeline = config.section("pipeline")
    stages = pipeline.get(
        "stages",
        ["render-batch", "convert", "sanity-check", "train", "evaluate"],
    )

    return "\n".join(
        [
            f"Config: {config.path}",
            f"Kitchen photo: {config.section('inputs').get('kitchen_photo', 'unset')}",
            (
                "Kitchen photo dir: "
                f"{config.section('inputs').get('kitchen_photo_dir', 'unset')}"
            ),
            (
                "Kitchen manifest: "
                f"{config.section('inputs').get('kitchen_manifest', 'unset')}"
            ),
            f"Layout spec: {config.section('inputs').get('layout_spec', 'unset')}",
            f"Layout diagnostics: {config.section('inputs').get('layout_diagnostics', 'unset')}",
            f"Layout preview: {config.section('inputs').get('layout_preview', 'unset')}",
            f"Render backend: {render.get('backend', 'unset')}",
            f"Batch output dir: {render.get('batch_output_dir', 'unset')}",
            f"Batch limit: {render.get('batch_limit', 'unset')}",
            f"Video length: {render.get('seconds', 'unset')}s @ {render.get('fps', 'unset')} fps",
            f"Pests: {', '.join(render.get('pest_types', [])) or 'unset'}",
            f"Frames dir: {dataset.get('frames_dir', 'unset')}",
            f"COCO annotations: {dataset.get('coco_annotations', 'unset')}",
            f"Training model: {training.get('model_name', 'unset')}",
            f"Training output: {training.get('output_dir', 'unset')}",
            f"Inference output: {inference.get('output_image', 'unset')}",
            f"Pipeline stages: {', '.join(stages) if isinstance(stages, list) else 'unset'}",
            f"DCC partition: {dcc.get('partition', 'unset')}",
        ]
    )
