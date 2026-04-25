"""Tests for dataset conversion outputs."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from prob_ml.dataset import convert_batch_render_outputs


class DummyConfig:
    """Small config stand-in for dataset conversion tests."""

    def __init__(self, repo_root: Path, raw: dict):
        self.repo_root = repo_root
        self.raw = raw

    def section(self, name: str) -> dict:
        return self.raw[name]


def _write_image(path: Path, *, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 120, 120)).save(path)


def _build_config(repo_root: Path) -> DummyConfig:
    return DummyConfig(
        repo_root=repo_root,
        raw={
            "inputs": {
                "kitchen_manifest": "data/raw/kitchen/metadata/manifest.csv",
            },
            "render": {
                "batch_output_dir": "artifacts/batch_render",
                "blender_script": "src/prob_ml/blender/render_scene.py",
            },
            "dataset": {
                "coco_annotations": "artifacts/dataset/coco_annotations.json",
            },
        },
    )


def test_convert_batch_render_outputs_builds_coco_and_yolo(tmp_path: Path) -> None:
    kitchen_dir = tmp_path / "data" / "raw" / "kitchen" / "images"
    _write_image(kitchen_dir / "train_bg.jpg", size=(200, 100))
    _write_image(kitchen_dir / "val_bg.jpg", size=(220, 120))
    _write_image(kitchen_dir / "neg_bg.jpg", size=(180, 90))

    manifest_path = tmp_path / "data" / "raw" / "kitchen" / "metadata" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,filename,relative_path,split,enabled",
                "kitchen_0001,train_bg.jpg,data/raw/kitchen/images/train_bg.jpg,train,true",
                "kitchen_0002,val_bg.jpg,data/raw/kitchen/images/val_bg.jpg,val,true",
                "kitchen_0003,neg_bg.jpg,data/raw/kitchen/images/neg_bg.jpg,neg_test,false",
            ]
        ),
        encoding="utf-8",
    )

    render_root = tmp_path / "artifacts" / "batch_render"
    train_frame = render_root / "kitchen_0001" / "frames" / "frame_00001.png"
    val_frame = render_root / "kitchen_0002" / "frames" / "frame_00001.png"
    _write_image(train_frame, size=(64, 32))
    _write_image(val_frame, size=(80, 40))

    train_annotations = [
        {
            "frame": 1,
            "file": str(train_frame.resolve()),
            "pests": [
                {
                    "label": "mouse",
                    "bbox": {
                        "x_min": 4.0,
                        "y_min": 6.0,
                        "x_max": 24.0,
                        "y_max": 20.0,
                        "width": 20.0,
                        "height": 14.0,
                    },
                }
            ],
        }
    ]
    val_annotations = [
        {
            "frame": 1,
            "file": str(val_frame.resolve()),
            "pests": [
                {
                    "label": "rat",
                    "bbox": {
                        "x_min": 10.0,
                        "y_min": 5.0,
                        "x_max": 34.0,
                        "y_max": 25.0,
                        "width": 24.0,
                        "height": 20.0,
                    },
                }
            ],
        }
    ]
    (render_root / "kitchen_0001").mkdir(parents=True, exist_ok=True)
    (render_root / "kitchen_0002").mkdir(parents=True, exist_ok=True)
    (render_root / "kitchen_0001" / "annotations.json").write_text(
        json.dumps(train_annotations),
        encoding="utf-8",
    )
    (render_root / "kitchen_0002" / "annotations.json").write_text(
        json.dumps(val_annotations),
        encoding="utf-8",
    )

    outputs = convert_batch_render_outputs(_build_config(tmp_path))

    coco_train = json.loads(outputs["coco_train"].read_text(encoding="utf-8"))
    coco_val = json.loads(outputs["coco_val"].read_text(encoding="utf-8"))
    coco_neg = json.loads(outputs["coco_neg_test"].read_text(encoding="utf-8"))
    neg_manifest = json.loads(outputs["neg_test_manifest"].read_text(encoding="utf-8"))
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    yolo_yaml = outputs["yolo_data"].read_text(encoding="utf-8")

    assert len(coco_train["images"]) == 1
    assert coco_train["annotations"][0]["category_id"] == 1
    assert (
        coco_train["images"][0]["file_name"]
        == "artifacts/batch_render/kitchen_0001/frames/frame_00001.png"
    )

    assert len(coco_val["images"]) == 1
    assert coco_val["annotations"][0]["category_id"] == 2

    assert len(coco_neg["images"]) == 1
    assert coco_neg["annotations"] == []
    assert neg_manifest["images"][0]["image_id"] == "kitchen_0003"

    assert "train" in summary["splits"]
    assert "val" in summary["splits"]
    assert summary["splits"]["neg_test"]["backgrounds"] == 1

    assert "train: images/train" in yolo_yaml
    assert "val: images/val" in yolo_yaml
    assert "test: images/neg_test" in yolo_yaml
    assert (tmp_path / "artifacts" / "dataset" / "yolo" / "labels" / "train").exists()
    train_label = (
        tmp_path
        / "artifacts"
        / "dataset"
        / "yolo"
        / "labels"
        / "train"
        / "kitchen_0001_frame_00001.txt"
    )
    assert train_label.exists()
    assert train_label.read_text(encoding="utf-8").startswith("0 ")


def test_convert_batch_render_outputs_rejects_invalid_bbox(tmp_path: Path) -> None:
    kitchen_dir = tmp_path / "data" / "raw" / "kitchen" / "images"
    _write_image(kitchen_dir / "train_bg.jpg", size=(100, 100))
    _write_image(kitchen_dir / "val_bg.jpg", size=(100, 100))

    manifest_path = tmp_path / "data" / "raw" / "kitchen" / "metadata" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,filename,relative_path,split,enabled",
                "kitchen_0001,train_bg.jpg,data/raw/kitchen/images/train_bg.jpg,train,true",
                "kitchen_0002,val_bg.jpg,data/raw/kitchen/images/val_bg.jpg,val,true",
            ]
        ),
        encoding="utf-8",
    )

    render_root = tmp_path / "artifacts" / "batch_render"
    train_frame = render_root / "kitchen_0001" / "frames" / "frame_00001.png"
    val_frame = render_root / "kitchen_0002" / "frames" / "frame_00001.png"
    _write_image(train_frame, size=(50, 50))
    _write_image(val_frame, size=(50, 50))
    (render_root / "kitchen_0001").mkdir(parents=True, exist_ok=True)
    (render_root / "kitchen_0002").mkdir(parents=True, exist_ok=True)

    (render_root / "kitchen_0001" / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "frame": 1,
                    "file": str(train_frame.resolve()),
                    "pests": [
                        {
                            "label": "mouse",
                            "bbox": {
                                "x_min": 45.0,
                                "y_min": 45.0,
                                "x_max": 60.0,
                                "y_max": 60.0,
                                "width": 15.0,
                                "height": 15.0,
                            },
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (render_root / "kitchen_0002" / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "frame": 1,
                    "file": str(val_frame.resolve()),
                    "pests": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    try:
        convert_batch_render_outputs(_build_config(tmp_path))
    except ValueError as exc:
        assert "Bbox exceeds image bounds" in str(exc)
    else:
        raise AssertionError("Expected invalid bbox conversion to fail")


def test_convert_batch_render_outputs_auto_assigns_train_val_from_rendered_unassigned(
    tmp_path: Path,
) -> None:
    kitchen_dir = tmp_path / "data" / "raw" / "kitchen" / "images"
    _write_image(kitchen_dir / "bg_1.jpg", size=(200, 100))
    _write_image(kitchen_dir / "bg_2.jpg", size=(220, 120))
    _write_image(kitchen_dir / "bg_3.jpg", size=(180, 90))

    manifest_path = tmp_path / "data" / "raw" / "kitchen" / "metadata" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,filename,relative_path,split,enabled",
                "kitchen_0001,bg_1.jpg,data/raw/kitchen/images/bg_1.jpg,unassigned,true",
                "kitchen_0002,bg_2.jpg,data/raw/kitchen/images/bg_2.jpg,unassigned,true",
                "kitchen_0003,bg_3.jpg,data/raw/kitchen/images/bg_3.jpg,unassigned,true",
            ]
        ),
        encoding="utf-8",
    )

    render_root = tmp_path / "artifacts" / "batch_render"
    frame_1 = render_root / "kitchen_0001" / "frames" / "frame_00001.png"
    frame_2 = render_root / "kitchen_0002" / "frames" / "frame_00001.png"
    _write_image(frame_1, size=(64, 32))
    _write_image(frame_2, size=(80, 40))
    (render_root / "kitchen_0001").mkdir(parents=True, exist_ok=True)
    (render_root / "kitchen_0002").mkdir(parents=True, exist_ok=True)
    (render_root / "kitchen_0001" / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "frame": 1,
                    "file": str(frame_1.resolve()),
                    "pests": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    (render_root / "kitchen_0002" / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "frame": 1,
                    "file": str(frame_2.resolve()),
                    "pests": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = convert_batch_render_outputs(_build_config(tmp_path))

    coco_train = json.loads(outputs["coco_train"].read_text(encoding="utf-8"))
    coco_val = json.loads(outputs["coco_val"].read_text(encoding="utf-8"))
    coco_neg = json.loads(outputs["coco_neg_test"].read_text(encoding="utf-8"))
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))

    assert len(coco_train["images"]) == 1
    assert len(coco_val["images"]) == 1
    assert len(coco_neg["images"]) == 1
    assert summary["splits"]["train"]["backgrounds_with_renders"] == 1
    assert summary["splits"]["val"]["backgrounds_with_renders"] == 1
    assert summary["splits"]["neg_test"]["backgrounds"] == 1


def test_convert_batch_render_outputs_recovers_from_stale_absolute_frame_path(
    tmp_path: Path,
) -> None:
    kitchen_dir = tmp_path / "data" / "raw" / "kitchen" / "images"
    _write_image(kitchen_dir / "train_bg.jpg", size=(200, 100))
    _write_image(kitchen_dir / "val_bg.jpg", size=(220, 120))

    manifest_path = tmp_path / "data" / "raw" / "kitchen" / "metadata" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,filename,relative_path,split,enabled",
                "kitchen_0001,train_bg.jpg,data/raw/kitchen/images/train_bg.jpg,train,true",
                "kitchen_0002,val_bg.jpg,data/raw/kitchen/images/val_bg.jpg,val,true",
            ]
        ),
        encoding="utf-8",
    )

    render_root = tmp_path / "artifacts" / "batch_render"
    train_frame = render_root / "kitchen_0001" / "frames" / "frame_00001.png"
    val_frame = render_root / "kitchen_0002" / "frames" / "frame_00001.png"
    _write_image(train_frame, size=(64, 32))
    _write_image(val_frame, size=(80, 40))
    (render_root / "kitchen_0001").mkdir(parents=True, exist_ok=True)
    (render_root / "kitchen_0002").mkdir(parents=True, exist_ok=True)

    stale_train_path = (
        "/hpc/home/hz365/prob_ml/artifacts/batch_render/"
        "kitchen_0001/frames/frame_00001.png"
    )
    stale_val_path = (
        "/hpc/home/hz365/prob_ml/artifacts/batch_render/"
        "kitchen_0002/frames/frame_00001.png"
    )
    (render_root / "kitchen_0001" / "annotations.json").write_text(
        json.dumps([{"frame": 1, "file": stale_train_path, "pests": []}]),
        encoding="utf-8",
    )
    (render_root / "kitchen_0002" / "annotations.json").write_text(
        json.dumps([{"frame": 1, "file": stale_val_path, "pests": []}]),
        encoding="utf-8",
    )

    outputs = convert_batch_render_outputs(_build_config(tmp_path))
    coco_train = json.loads(outputs["coco_train"].read_text(encoding="utf-8"))
    coco_val = json.loads(outputs["coco_val"].read_text(encoding="utf-8"))

    assert len(coco_train["images"]) == 1
    assert len(coco_val["images"]) == 1
