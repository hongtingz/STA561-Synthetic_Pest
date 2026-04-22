"""Tests for layout-spec generation."""

from __future__ import annotations

import json
from pathlib import Path

from prob_ml.layout import (
    PhotoCuesSpec,
    build_layout_spec,
    save_layout_diagnostics,
    save_layout_spec,
)
from prob_ml.render import build_blender_command


class DummyConfig:
    """Small config stand-in for render command tests."""

    def __init__(self, repo_root: Path, raw: dict):
        self.repo_root = repo_root
        self.raw = raw

    def section(self, name: str) -> dict:
        return self.raw[name]


def test_build_layout_spec_includes_requested_pests(tmp_path: Path) -> None:
    photo_path = tmp_path / "kitchen.jpg"
    photo_path.write_bytes(b"fake")
    photo_cues = PhotoCuesSpec(
        brightness_top=0.8,
        brightness_mid=0.55,
        brightness_bottom=0.35,
        left_brightness=0.45,
        center_brightness=0.52,
        right_brightness=0.62,
        floor_line_ratio=0.7,
        clutter_score=0.3,
        warm_bias=0.6,
        contrast_score=0.5,
    )

    layout = build_layout_spec(
        photo_path=photo_path,
        photo_size=(1280, 720),
        photo_cues=photo_cues,
        pest_types=["mouse", "rat", "cockroach"],
        scene_seed=42,
    )

    pest_types = [pest.pest_type for pest in layout.pests]
    assert pest_types == ["mouse", "rat", "cockroach"]
    assert layout.room.width > 0
    assert layout.room.depth > 0
    assert layout.photo_cues.floor_line_ratio == 0.7


def test_save_layout_spec_writes_json(tmp_path: Path) -> None:
    photo_path = tmp_path / "kitchen.jpg"
    photo_path.write_bytes(b"fake")
    photo_cues = PhotoCuesSpec(
        brightness_top=0.75,
        brightness_mid=0.5,
        brightness_bottom=0.4,
        left_brightness=0.3,
        center_brightness=0.5,
        right_brightness=0.7,
        floor_line_ratio=0.68,
        clutter_score=0.35,
        warm_bias=0.58,
        contrast_score=0.4,
    )
    layout = build_layout_spec(
        photo_path=photo_path,
        photo_size=(640, 480),
        photo_cues=photo_cues,
        pest_types=["mouse"],
        scene_seed=7,
    )
    output_path = tmp_path / "layout.json"

    save_layout_spec(layout, output_path)

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw["source_photo"] == str(photo_path)
    assert raw["pests"][0]["pest_type"] == "mouse"
    assert "photo_cues" in raw


def test_save_layout_diagnostics_writes_summary(tmp_path: Path) -> None:
    photo_path = tmp_path / "kitchen.jpg"
    photo_path.write_bytes(b"fake")
    photo_cues = PhotoCuesSpec(
        brightness_top=0.82,
        brightness_mid=0.61,
        brightness_bottom=0.47,
        left_brightness=0.41,
        center_brightness=0.58,
        right_brightness=0.67,
        floor_line_ratio=0.74,
        clutter_score=0.29,
        warm_bias=0.64,
        contrast_score=0.51,
    )
    layout = build_layout_spec(
        photo_path=photo_path,
        photo_size=(1280, 720),
        photo_cues=photo_cues,
        pest_types=["mouse", "rat"],
        scene_seed=11,
    )
    output_path = tmp_path / "layout_diagnostics.json"

    save_layout_diagnostics(layout, output_path)

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw["fixture_summary"]["fixture_count"] >= 1
    assert raw["pest_types"] == ["mouse", "rat"]


def test_build_blender_command_contains_layout_and_outputs(tmp_path: Path) -> None:
    config = DummyConfig(
        repo_root=tmp_path,
        raw={
            "inputs": {
                "kitchen_photo": "data/inputs/kitchen.jpg",
                "layout_spec": "data/intermediate/layout/layout.json",
            },
            "render": {
                "backend": "blender",
                "blender_executable": "blender",
                "blender_script": "src/prob_ml/blender/render_scene.py",
                "fps": 30,
                "seconds": 30,
                "resolution_width": 1280,
                "resolution_height": 720,
                "samples": 64,
            },
            "dataset": {
                "frames_dir": "artifacts/render/frames",
                "annotations_raw": "artifacts/render/annotations.json",
            },
        },
    )
    resolved_paths = {
        "layout_spec": tmp_path / "data/intermediate/layout/layout.json",
        "frames_dir": tmp_path / "artifacts/render/frames",
        "annotations": tmp_path / "artifacts/render/annotations.json",
        "blender_script": tmp_path / "src/prob_ml/blender/render_scene.py",
    }

    command = build_blender_command(config, resolved_paths)

    assert command[0] == "blender"
    assert "--layout-spec" in command
    assert str(resolved_paths["layout_spec"]) in command


def test_photo_cues_change_layout_geometry(tmp_path: Path) -> None:
    photo_path = tmp_path / "kitchen.jpg"
    photo_path.write_bytes(b"fake")

    bright_open = PhotoCuesSpec(
        brightness_top=0.9,
        brightness_mid=0.75,
        brightness_bottom=0.7,
        left_brightness=0.7,
        center_brightness=0.72,
        right_brightness=0.68,
        floor_line_ratio=0.8,
        clutter_score=0.15,
        warm_bias=0.62,
        contrast_score=0.55,
    )
    dark_cluttered = PhotoCuesSpec(
        brightness_top=0.35,
        brightness_mid=0.28,
        brightness_bottom=0.2,
        left_brightness=0.22,
        center_brightness=0.3,
        right_brightness=0.4,
        floor_line_ratio=0.56,
        clutter_score=0.78,
        warm_bias=0.42,
        contrast_score=0.24,
    )

    open_layout = build_layout_spec(
        photo_path=photo_path,
        photo_size=(1280, 720),
        photo_cues=bright_open,
        pest_types=["mouse"],
        scene_seed=3,
    )
    cluttered_layout = build_layout_spec(
        photo_path=photo_path,
        photo_size=(1280, 720),
        photo_cues=dark_cluttered,
        pest_types=["mouse"],
        scene_seed=3,
    )

    assert open_layout.room.depth > cluttered_layout.room.depth
    assert open_layout.camera.lens_mm > cluttered_layout.camera.lens_mm
    assert open_layout.lights[0].energy > cluttered_layout.lights[0].energy
