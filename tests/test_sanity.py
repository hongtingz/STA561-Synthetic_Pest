"""Tests for dataset sanity-check helpers."""

from __future__ import annotations

import json
from pathlib import Path

from prob_ml.sanity import inspect_coco_file


def _write_coco(path: Path, bbox: list[float]) -> None:
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "frame.jpg",
                        "width": 100,
                        "height": 80,
                        "background_image_id": "kitchen_001",
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "mouse"}],
            }
        ),
        encoding="utf-8",
    )


def test_inspect_coco_file_reports_class_counts(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"placeholder")
    coco_path = tmp_path / "coco.json"
    _write_coco(coco_path, [1, 2, 10, 12])

    report = inspect_coco_file(coco_path, tmp_path, split="train")

    assert report["images"] == 1
    assert report["annotations"] == 1
    assert report["class_counts"] == {"mouse": 1}
    assert report["errors"] == []


def test_inspect_coco_file_catches_out_of_bounds_bbox(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"placeholder")
    coco_path = tmp_path / "coco.json"
    _write_coco(coco_path, [95, 2, 10, 12])

    report = inspect_coco_file(coco_path, tmp_path, split="train")

    assert any("exceeds width" in error for error in report["errors"])
