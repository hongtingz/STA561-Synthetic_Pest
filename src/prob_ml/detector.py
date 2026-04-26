"""Shared detector training, inference, and evaluation helpers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORY_ID_TO_NAME = {
    1: "mouse",
    2: "rat",
    3: "cockroach",
}
NUM_DETECTOR_CLASSES = len(CATEGORY_ID_TO_NAME) + 1
TRANSFORMER_DETECTOR_CLASSES = len(CATEGORY_ID_TO_NAME)
YOLOS_TINY_MODEL_ID = "hustvl/yolos-tiny"
DETECTOR_MODEL_ALIASES = {
    "vit": YOLOS_TINY_MODEL_ID,
    "vit_detector": YOLOS_TINY_MODEL_ID,
    "yolos": YOLOS_TINY_MODEL_ID,
    "yolos-tiny": YOLOS_TINY_MODEL_ID,
    "yolos_tiny": YOLOS_TINY_MODEL_ID,
}


@dataclass
class MatchSummary:
    """Simple detection summary for project-level TDR/FPR reporting."""

    evaluated_images: int
    ground_truth_boxes: int
    predicted_boxes: int
    matched_boxes: int
    false_positive_images: int
    per_class_ground_truth: dict[str, int]
    per_class_matched: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        tdr = self.matched_boxes / self.ground_truth_boxes if self.ground_truth_boxes else 0.0
        fpr = (
            self.false_positive_images / self.evaluated_images
            if self.evaluated_images
            else 0.0
        )
        per_class_detection_rate = {}
        for class_name, count in self.per_class_ground_truth.items():
            matched = self.per_class_matched.get(class_name, 0)
            per_class_detection_rate[class_name] = matched / count if count else 0.0

        return {
            "evaluated_images": self.evaluated_images,
            "ground_truth_boxes": self.ground_truth_boxes,
            "predicted_boxes": self.predicted_boxes,
            "matched_boxes": self.matched_boxes,
            "false_positive_images": self.false_positive_images,
            "true_detection_rate": tdr,
            "false_positive_rate": fpr,
            "per_class_ground_truth": self.per_class_ground_truth,
            "per_class_matched": self.per_class_matched,
            "per_class_detection_rate": per_class_detection_rate,
        }


def resolve_repo_path(repo_root: Path, raw_path: str | Path) -> Path:
    """Resolve an absolute or repo-relative path."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_coco_detection_json(path: Path) -> dict[str, Any]:
    """Load a COCO-style detection JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "images" not in payload or "annotations" not in payload:
        raise ValueError(f"COCO file must contain images and annotations: {path}")
    return payload


def xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    """Convert COCO [x, y, width, height] to [x1, y1, x2, y2]."""
    x, y, width, height = [float(value) for value in box]
    return [x, y, x + width, y + height]


def box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU for two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_prediction_to_target(
    prediction: dict[str, Any],
    target: dict[str, Any],
    *,
    score_threshold: float,
    iou_threshold: float,
) -> MatchSummary:
    """Match one image prediction to one target using greedy same-class IoU matching."""
    gt_boxes = target.get("boxes", [])
    gt_labels = target.get("labels", [])
    pred_boxes = prediction.get("boxes", [])
    pred_labels = prediction.get("labels", [])
    pred_scores = prediction.get("scores", [])

    kept_predictions = [
        (index, float(score))
        for index, score in enumerate(pred_scores)
        if float(score) >= score_threshold
    ]
    kept_predictions.sort(key=lambda item: item[1], reverse=True)

    matched_gt: set[int] = set()
    matched_boxes = 0
    per_class_ground_truth: dict[str, int] = defaultdict(int)
    per_class_matched: dict[str, int] = defaultdict(int)

    for label in gt_labels:
        per_class_ground_truth[CATEGORY_ID_TO_NAME.get(int(label), str(label))] += 1

    for pred_index, _ in kept_predictions:
        pred_label = int(pred_labels[pred_index])
        pred_box = [float(value) for value in pred_boxes[pred_index]]
        best_gt_index: int | None = None
        best_iou = 0.0
        for gt_index, gt_box_raw in enumerate(gt_boxes):
            if gt_index in matched_gt or int(gt_labels[gt_index]) != pred_label:
                continue
            gt_box = [float(value) for value in gt_box_raw]
            iou = box_iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_index)
            matched_boxes += 1
            class_name = CATEGORY_ID_TO_NAME.get(pred_label, str(pred_label))
            per_class_matched[class_name] += 1

    return MatchSummary(
        evaluated_images=1,
        ground_truth_boxes=len(gt_boxes),
        predicted_boxes=len(kept_predictions),
        matched_boxes=matched_boxes,
        false_positive_images=1 if not gt_boxes and kept_predictions else 0,
        per_class_ground_truth=dict(per_class_ground_truth),
        per_class_matched=dict(per_class_matched),
    )


def combine_match_summaries(summaries: list[MatchSummary]) -> MatchSummary:
    """Combine per-image detection summaries."""
    per_class_ground_truth: dict[str, int] = defaultdict(int)
    per_class_matched: dict[str, int] = defaultdict(int)
    for summary in summaries:
        for class_name, count in summary.per_class_ground_truth.items():
            per_class_ground_truth[class_name] += count
        for class_name, count in summary.per_class_matched.items():
            per_class_matched[class_name] += count

    return MatchSummary(
        evaluated_images=sum(summary.evaluated_images for summary in summaries),
        ground_truth_boxes=sum(summary.ground_truth_boxes for summary in summaries),
        predicted_boxes=sum(summary.predicted_boxes for summary in summaries),
        matched_boxes=sum(summary.matched_boxes for summary in summaries),
        false_positive_images=sum(summary.false_positive_images for summary in summaries),
        per_class_ground_truth=dict(per_class_ground_truth),
        per_class_matched=dict(per_class_matched),
    )


class CocoDetectionDataset:
    """Small COCO detection dataset wrapper for torchvision detectors."""

    def __init__(
        self,
        coco_path: Path,
        repo_root: Path,
        *,
        max_images: int | None = None,
        augment: dict[str, Any] | None = None,
    ):
        self.coco_path = coco_path
        self.repo_root = repo_root
        self.augment = augment or {}
        self.payload = load_coco_detection_json(coco_path)
        images = list(self.payload["images"])
        if max_images is not None:
            images = images[:max_images]
        self.images = images

        annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.payload["annotations"]:
            annotations_by_image_id[int(annotation["image_id"])].append(annotation)
        self.annotations_by_image_id = annotations_by_image_id

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import functional as transform_functional

        image_info = self.images[index]
        image_path = resolve_repo_path(self.repo_root, str(image_info["file_name"]))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            color_jitter = self.augment.get("color_jitter", {})
            if color_jitter:
                jitter = transforms.ColorJitter(
                    brightness=float(color_jitter.get("brightness", 0.0)),
                    contrast=float(color_jitter.get("contrast", 0.0)),
                    saturation=float(color_jitter.get("saturation", 0.0)),
                    hue=float(color_jitter.get("hue", 0.0)),
                )
                image = jitter(image)
            image_tensor = transform_functional.to_tensor(image)

        annotations = self.annotations_by_image_id.get(int(image_info["id"]), [])
        boxes = [xywh_to_xyxy(annotation["bbox"]) for annotation in annotations]
        labels = [int(annotation["category_id"]) for annotation in annotations]
        area = [float(annotation.get("area", 0.0)) for annotation in annotations]
        iscrowd = [int(annotation.get("iscrowd", 0)) for annotation in annotations]
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape((-1, 4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([int(image_info["id"])], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        flip_probability = float(self.augment.get("horizontal_flip_probability", 0.0))
        if flip_probability and random.random() < flip_probability:
            image_tensor = transform_functional.hflip(image_tensor)
            image_width = float(image_info["width"])
            boxes_tensor = target["boxes"]
            if boxes_tensor.numel():
                flipped_boxes = boxes_tensor.clone()
                flipped_boxes[:, 0] = image_width - boxes_tensor[:, 2]
                flipped_boxes[:, 2] = image_width - boxes_tensor[:, 0]
                target["boxes"] = flipped_boxes
        return image_tensor, target


def collate_detection_batch(batch):
    """Torchvision detection models expect lists of images and targets."""
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def select_device(raw_device: str):
    """Resolve auto/cuda/cpu device strings."""
    import torch

    device = raw_device.lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return torch.device(device)


def normalize_detector_model_name(model_name: str) -> str:
    """Normalize detector aliases to a canonical model identifier."""
    normalized = str(model_name).strip()
    if normalized.startswith("hf:"):
        normalized = normalized[3:]
    return DETECTOR_MODEL_ALIASES.get(normalized, normalized)


def is_transformer_detector(model_name: str) -> bool:
    """Return True when a detector model is transformer-based."""
    return normalize_detector_model_name(model_name) == YOLOS_TINY_MODEL_ID


def _xyxy_to_cxcywh_tensor(boxes):
    import torch

    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
    converted = boxes.clone()
    converted[:, 2] = boxes[:, 2] - boxes[:, 0]
    converted[:, 3] = boxes[:, 3] - boxes[:, 1]
    converted[:, 0] = boxes[:, 0] + converted[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] + converted[:, 3] / 2.0
    return converted


def _cxcywh_to_xyxy_tensor(boxes):
    import torch

    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
    converted = boxes.clone()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def prepare_transformer_training_batch(
    images,
    targets,
    device,
    *,
    image_size: int,
):
    """Resize images and convert COCO-style targets to YOLOS training labels."""
    import torch
    from torchvision.transforms.functional import resize

    resized_images = []
    labels = []
    for image_tensor, target in zip(images, targets, strict=True):
        original_height = float(image_tensor.shape[-2])
        original_width = float(image_tensor.shape[-1])
        resized_image = resize(image_tensor, [image_size, image_size], antialias=True)
        resized_images.append(resized_image)

        boxes = target["boxes"].detach().clone()
        if boxes.numel():
            boxes[:, [0, 2]] *= float(image_size) / original_width
            boxes[:, [1, 3]] *= float(image_size) / original_height
            boxes = boxes / float(image_size)
            boxes = _xyxy_to_cxcywh_tensor(boxes)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        class_labels = target["labels"].detach().clone() - 1
        labels.append(
            {
                "class_labels": class_labels.to(device),
                "boxes": boxes.to(device),
            }
        )

    pixel_values = torch.stack(resized_images).to(device)
    return pixel_values, labels


def predict_transformer_batch(
    model,
    images,
    device,
    *,
    image_size: int,
):
    """Run a transformer detector and convert outputs to project prediction dicts."""
    import torch
    from torchvision.transforms.functional import resize

    resized_images = []
    original_sizes: list[tuple[int, int]] = []
    for image_tensor in images:
        original_sizes.append((int(image_tensor.shape[-2]), int(image_tensor.shape[-1])))
        resized_images.append(resize(image_tensor, [image_size, image_size], antialias=True))

    pixel_values = torch.stack(resized_images).to(device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.detach().cpu()
    pred_boxes = outputs.pred_boxes.detach().cpu()

    predictions = []
    for batch_logits, batch_boxes, (height, width) in zip(
        logits,
        pred_boxes,
        original_sizes,
        strict=True,
    ):
        probabilities = batch_logits.softmax(-1)
        scores, labels = probabilities[..., :-1].max(-1)
        boxes_xyxy = _cxcywh_to_xyxy_tensor(batch_boxes)
        boxes_xyxy[:, [0, 2]] *= float(width)
        boxes_xyxy[:, [1, 3]] *= float(height)
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0.0, float(width))
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0.0, float(height))
        predictions.append(
            {
                "boxes": boxes_xyxy.tolist(),
                "labels": (labels + 1).tolist(),
                "scores": scores.tolist(),
            }
        )
    return predictions


def build_detection_model(
    model_name: str,
    *,
    num_classes: int = NUM_DETECTOR_CLASSES,
    pretrained: bool = False,
):
    """Build a detector model for the project training and evaluation pipeline."""
    normalized_name = normalize_detector_model_name(model_name)

    if normalized_name == YOLOS_TINY_MODEL_ID:
        from transformers import AutoConfig, AutoModelForObjectDetection

        id2label = {
            0: "mouse",
            1: "rat",
            2: "cockroach",
        }
        label2id = {label: index for index, label in id2label.items()}
        if pretrained:
            return AutoModelForObjectDetection.from_pretrained(
                normalized_name,
                num_labels=TRANSFORMER_DETECTOR_CLASSES,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

        config = AutoConfig.from_pretrained(normalized_name)
        config.num_labels = TRANSFORMER_DETECTOR_CLASSES
        config.id2label = id2label
        config.label2id = label2id
        return AutoModelForObjectDetection.from_config(config)

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if normalized_name != "fasterrcnn_mobilenet_v3_large_320_fpn":
        raise ValueError(
            "Supported detector models are fasterrcnn_mobilenet_v3_large_320_fpn "
            f"and {YOLOS_TINY_MODEL_ID!r}, got {model_name!r}."
        )

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def move_targets_to_device(targets, device):
    """Move torchvision target dictionaries to a device."""
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def tensor_prediction_to_python(prediction: dict[str, Any]) -> dict[str, Any]:
    """Convert a torchvision prediction dict to plain Python lists."""
    return {
        "boxes": prediction["boxes"].detach().cpu().tolist(),
        "labels": prediction["labels"].detach().cpu().tolist(),
        "scores": prediction["scores"].detach().cpu().tolist(),
    }


def tensor_target_to_python(target: dict[str, Any]) -> dict[str, Any]:
    """Convert a torchvision target dict to plain Python lists."""
    return {
        "boxes": target["boxes"].detach().cpu().tolist(),
        "labels": target["labels"].detach().cpu().tolist(),
    }


def checkpoint_payload(
    model,
    *,
    model_name: str,
    metrics: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable detector checkpoint payload."""
    normalized_name = normalize_detector_model_name(model_name)
    payload = {
        "model_name": normalized_name,
        "num_classes": (
            TRANSFORMER_DETECTOR_CLASSES
            if is_transformer_detector(normalized_name)
            else NUM_DETECTOR_CLASSES
        ),
        "categories": CATEGORY_ID_TO_NAME,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if extra_state:
        payload.update(extra_state)
    return payload
