"""Tests for layout-spec generation."""

from __future__ import annotations

import json
from pathlib import Path

from prob_ml.layout import build_layout_spec, save_layout_spec
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

    layout = build_layout_spec(
        photo_path=photo_path,
        photo_size=(1280, 720),
        pest_types=["mouse", "rat", "cockroach"],
        scene_seed=42,
    )

    pest_types = [pest.pest_type for pest in layout.pests]
    assert pest_types == ["mouse", "rat", "cockroach"]
    assert layout.room.width > 0
    assert layout.room.depth > 0


def test_save_layout_spec_writes_json(tmp_path: Path) -> None:
    photo_path = tmp_path / "kitchen.jpg"
    photo_path.write_bytes(b"fake")
    layout = build_layout_spec(
        photo_path=photo_path,
        photo_size=(640, 480),
        pest_types=["mouse"],
        scene_seed=7,
    )
    output_path = tmp_path / "layout.json"

    save_layout_spec(layout, output_path)

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw["source_photo"] == str(photo_path)
    assert raw["pests"][0]["pest_type"] == "mouse"


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
