"""Tests for detector helper logic that does not require torch."""

from __future__ import annotations

import json
from pathlib import Path

from prob_ml.detector import (
    CocoDetectionDataset,
    box_iou_xyxy,
    combine_match_summaries,
    match_prediction_to_target,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy_and_iou() -> None:
    assert xywh_to_xyxy([10, 20, 30, 40]) == [10.0, 20.0, 40.0, 60.0]
    assert box_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_match_prediction_to_target_counts_true_detection() -> None:
    prediction = {
        "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
        "labels": [1, 2],
        "scores": [0.95, 0.9],
    }
    target = {
        "boxes": [[0, 0, 10, 10]],
        "labels": [1],
    }

    summary = match_prediction_to_target(
        prediction,
        target,
        score_threshold=0.5,
        iou_threshold=0.5,
    )

    assert summary.ground_truth_boxes == 1
    assert summary.predicted_boxes == 2
    assert summary.matched_boxes == 1
    assert summary.to_dict()["true_detection_rate"] == 1.0


def test_match_prediction_to_target_counts_negative_false_positive() -> None:
    prediction = {
        "boxes": [[0, 0, 10, 10]],
        "labels": [1],
        "scores": [0.8],
    }
    target = {"boxes": [], "labels": []}

    summary = match_prediction_to_target(
        prediction,
        target,
        score_threshold=0.5,
        iou_threshold=0.5,
    )

    assert summary.false_positive_images == 1
    assert summary.to_dict()["false_positive_rate"] == 1.0


def test_combine_match_summaries() -> None:
    positive = match_prediction_to_target(
        {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
        {"boxes": [[0, 0, 10, 10]], "labels": [1]},
        score_threshold=0.5,
        iou_threshold=0.5,
    )
    negative = match_prediction_to_target(
        {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
        {"boxes": [], "labels": []},
        score_threshold=0.5,
        iou_threshold=0.5,
    )

    combined = combine_match_summaries([positive, negative]).to_dict()

    assert combined["evaluated_images"] == 2
    assert combined["matched_boxes"] == 1
    assert combined["false_positive_images"] == 1


def test_coco_detection_dataset_indexes_images_without_torch(tmp_path: Path) -> None:
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "frames/frame_00001.png",
                        "width": 64,
                        "height": 64,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 2, 10, 12],
                        "area": 120,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "mouse"}],
            }
        ),
        encoding="utf-8",
    )

    dataset = CocoDetectionDataset(coco_path, tmp_path)

    assert len(dataset) == 1
    assert dataset.annotations_by_image_id[1][0]["category_id"] == 1
