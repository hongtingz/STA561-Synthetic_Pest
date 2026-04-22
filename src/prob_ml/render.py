"""Rendering entrypoints for Blender-based synthetic generation."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from prob_ml.config import PipelineConfig
from prob_ml.layout import (
    build_layout_spec,
    extract_photo_cues,
    save_layout_diagnostics,
    save_layout_spec,
    summarize_layout_decisions,
)
from prob_ml.manifest import KitchenPhotoRecord, load_kitchen_photo_manifest
from prob_ml.preview import save_layout_preview


def _resolve_path(config: PipelineConfig, relative_path: str) -> Path:
    """Resolve a repo-relative path from the config."""
    return (config.repo_root / relative_path).resolve()


def _coerce_int(value: object, *, default: int | None = None) -> int | None:
    if value in {None, ""}:
        return default
    return int(value)


def _load_photo_size(photo_path: Path) -> tuple[int, int]:
    """Read the input photo dimensions."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required to infer the layout spec from the kitchen photo."
        ) from exc

    with Image.open(photo_path) as image:
        width, height = image.size
    return int(width), int(height)


def resolve_render_paths(config: PipelineConfig) -> dict[str, Path]:
    """Resolve key file and directory paths for the render stage."""
    inputs = config.section("inputs")
    render = config.section("render")
    dataset = config.section("dataset")
    return {
        "photo": _resolve_path(config, inputs["kitchen_photo"]),
        "layout_spec": _resolve_path(config, inputs["layout_spec"]),
        "layout_diagnostics": _resolve_path(
            config,
            inputs.get("layout_diagnostics", "artifacts/layout/layout_diagnostics.json"),
        ),
        "layout_preview": _resolve_path(
            config,
            inputs.get("layout_preview", "artifacts/layout/layout_preview.svg"),
        ),
        "frames_dir": _resolve_path(config, dataset["frames_dir"]),
        "annotations": _resolve_path(config, dataset["annotations_raw"]),
        "blender_script": _resolve_path(
            config,
            render.get("blender_script", "src/prob_ml/blender/render_scene.py"),
        ),
    }


def resolve_batch_render_paths(
    config: PipelineConfig, record: KitchenPhotoRecord
) -> dict[str, Path]:
    """Resolve per-image output paths for batch rendering."""
    render = config.section("render")
    sample_root = _resolve_path(
        config,
        render.get("batch_output_dir", "artifacts/batch_render"),
    ) / record.image_id
    return {
        "photo": record.photo_path,
        "layout_spec": sample_root / "layout.json",
        "layout_diagnostics": sample_root / "layout_diagnostics.json",
        "layout_preview": sample_root / "layout_preview.svg",
        "frames_dir": sample_root / "frames",
        "annotations": sample_root / "annotations.json",
        "blender_script": _resolve_path(
            config,
            render.get("blender_script", "src/prob_ml/blender/render_scene.py"),
        ),
    }


def _ensure_render_output_dirs(resolved_paths: dict[str, Path]) -> None:
    for key in [
        "layout_spec",
        "layout_diagnostics",
        "layout_preview",
        "frames_dir",
        "annotations",
    ]:
        path = resolved_paths[key]
        target_dir = path if path.suffix == "" else path.parent
        target_dir.mkdir(parents=True, exist_ok=True)


def prepare_layout_outputs(
    config: PipelineConfig,
    photo_path: Path,
    resolved_paths: dict[str, Path],
    *,
    scene_seed: int,
) -> tuple[Path, dict[str, object]]:
    """Generate layout artifacts for a specific kitchen photo."""
    render = config.section("render")
    if not photo_path.exists():
        raise FileNotFoundError(
            f"Kitchen photo not found: {photo_path}. Add an input image before running render."
        )

    _ensure_render_output_dirs(resolved_paths)
    photo_size = _load_photo_size(photo_path)
    photo_cues = extract_photo_cues(photo_path)
    layout_spec = build_layout_spec(
        photo_path=photo_path,
        photo_size=photo_size,
        photo_cues=photo_cues,
        pest_types=render.get("pest_types", []),
        scene_seed=scene_seed,
    )
    save_layout_spec(layout_spec, resolved_paths["layout_spec"])
    save_layout_diagnostics(layout_spec, resolved_paths["layout_diagnostics"])
    save_layout_preview(layout_spec, resolved_paths["layout_preview"])
    return resolved_paths["layout_spec"], summarize_layout_decisions(layout_spec)


def ensure_layout_spec(config: PipelineConfig, resolved_paths: dict[str, Path]) -> Path:
    """Generate and persist a layout spec from the configured kitchen photo."""
    layout_spec_path, _ = prepare_layout_outputs(
        config,
        resolved_paths["photo"],
        resolved_paths,
        scene_seed=int(config.section("render").get("scene_seed", 42)),
    )
    return layout_spec_path


def build_blender_command(config: PipelineConfig, resolved_paths: dict[str, Path]) -> list[str]:
    """Create the Blender command used for synthetic rendering."""
    render = config.section("render")
    return [
        str(render.get("blender_executable", "blender")),
        "--background",
        "--python",
        str(resolved_paths["blender_script"]),
        "--",
        "--layout-spec",
        str(resolved_paths["layout_spec"]),
        "--frames-dir",
        str(resolved_paths["frames_dir"]),
        "--annotations",
        str(resolved_paths["annotations"]),
        "--fps",
        str(int(render.get("fps", 30))),
        "--seconds",
        str(int(render.get("seconds", 30))),
        "--width",
        str(int(render.get("resolution_width", 1280))),
        "--height",
        str(int(render.get("resolution_height", 720))),
        "--samples",
        str(int(render.get("samples", 64))),
        "--render-device",
        str(render.get("render_device", "CPU")).upper(),
        "--compute-backend",
        str(render.get("compute_backend", "AUTO")).upper(),
    ]


def run_render(config: PipelineConfig) -> None:
    """Generate a layout spec and prepare the Blender render command."""
    render = config.section("render")
    resolved_paths = resolve_render_paths(config)
    layout_spec_path, diagnostics = prepare_layout_outputs(
        config,
        resolved_paths["photo"],
        resolved_paths,
        scene_seed=int(render.get("scene_seed", 42)),
    )
    command = build_blender_command(config, resolved_paths)

    print("Render stage")
    print(f"  backend={render.get('backend')}")
    print(f"  kitchen_photo={resolved_paths['photo']}")
    print(f"  layout_spec={layout_spec_path}")
    print(f"  layout_diagnostics={resolved_paths['layout_diagnostics']}")
    print(f"  layout_preview={resolved_paths['layout_preview']}")
    print(f"  blender_script={resolved_paths['blender_script']}")
    print(f"  frames_dir={resolved_paths['frames_dir']}")
    print(f"  annotations={resolved_paths['annotations']}")
    print(f"  pests={render.get('pest_types')}")
    print(f"  render_device={str(render.get('render_device', 'CPU')).upper()}")
    print(f"  compute_backend={str(render.get('compute_backend', 'AUTO')).upper()}")
    print("  layout_cues=content-driven from kitchen photo")
    print(
        "  layout_summary="
        f"room=({diagnostics['room']['width']}x"
        f"{diagnostics['room']['depth']}x{diagnostics['room']['height']}), "
        f"fridge_side={diagnostics['fixture_summary']['fridge_side']}, "
        f"lens={diagnostics['camera']['lens_mm']}mm"
    )
    print("  blender_command=")
    print("    " + " ".join(shlex.quote(part) for part in command))

    if render.get("execute", False):
        subprocess.run(command, check=True)


def run_render_batch(config: PipelineConfig) -> None:
    """Prepare or execute per-image renders from the kitchen-photo manifest."""
    render = config.section("render")
    inputs = config.section("inputs")
    manifest_path = _resolve_path(config, inputs["kitchen_manifest"])
    batch_limit = _coerce_int(render.get("batch_limit"))
    enabled_only = bool(render.get("batch_enabled_only", True))
    records = load_kitchen_photo_manifest(
        manifest_path,
        config.repo_root,
        limit=batch_limit,
        enabled_only=enabled_only,
    )
    if not records:
        raise RuntimeError(
            f"No kitchen photos selected from manifest: {manifest_path}"
        )

    batch_root = _resolve_path(
        config,
        render.get("batch_output_dir", "artifacts/batch_render"),
    )
    print("Render batch stage")
    print(f"  manifest={manifest_path}")
    print(f"  batch_output_dir={batch_root}")
    print(f"  selected_images={len(records)}")
    print(f"  execute={bool(render.get('execute', False))}")

    base_seed = int(render.get("scene_seed", 42))
    for index, record in enumerate(records, start=1):
        resolved_paths = resolve_batch_render_paths(config, record)
        layout_spec_path, diagnostics = prepare_layout_outputs(
            config,
            record.photo_path,
            resolved_paths,
            scene_seed=base_seed + index - 1,
        )
        command = build_blender_command(config, resolved_paths)

        print(f"[{index}/{len(records)}] image_id={record.image_id}")
        print(f"  split={record.split}")
        print(f"  photo={record.photo_path}")
        print(f"  layout_spec={layout_spec_path}")
        print(f"  frames_dir={resolved_paths['frames_dir']}")
        print(
            "  layout_summary="
            f"room=({diagnostics['room']['width']}x"
            f"{diagnostics['room']['depth']}x{diagnostics['room']['height']}), "
            f"fridge_side={diagnostics['fixture_summary']['fridge_side']}, "
            f"lens={diagnostics['camera']['lens_mm']}mm"
        )
        print("  blender_command=")
        print("    " + " ".join(shlex.quote(part) for part in command))

        if render.get("execute", False):
            subprocess.run(command, check=True)
