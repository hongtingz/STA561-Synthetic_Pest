"""Checkpoint evaluation utilities for detector experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prob_ml.config import PipelineConfig
from prob_ml.detector import (
    CATEGORY_ID_TO_NAME,
    CocoDetectionDataset,
    build_detection_model,
    collate_detection_batch,
    is_transformer_detector,
    match_prediction_to_target,
    predict_transformer_batch,
    resolve_repo_path,
    select_device,
    tensor_prediction_to_python,
    tensor_target_to_python,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_loader(dataset):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_detection_batch,
    )


def _draw_eval_example(
    image_path: Path,
    output_path: Path,
    prediction: dict[str, Any],
    target: dict[str, Any],
    *,
    threshold: float,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except OSError:
            font = None

        for box, label in zip(target["boxes"], target["labels"], strict=True):
            x1, y1, x2, y2 = [float(value) for value in box]
            text = f"gt:{CATEGORY_ID_TO_NAME.get(int(label), str(label))}"
            draw.rectangle([x1, y1, x2, y2], outline=(40, 180, 90), width=3)
            draw.text((x1, y1), text, fill=(40, 180, 90), font=font)

        for box, label, score in zip(
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
            strict=True,
        ):
            if float(score) < threshold:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            text = f"pred:{CATEGORY_ID_TO_NAME.get(int(label), str(label))} {score:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(240, 60, 40), width=3)
            draw.text((x1, max(0.0, y1 - 12.0)), text, fill=(240, 60, 40), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def _summarize_pairs(
    pairs: list[dict[str, Any]],
    *,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    from prob_ml.detector import combine_match_summaries

    summaries = [
        match_prediction_to_target(
            pair["prediction"],
            pair["target"],
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        for pair in pairs
    ]
    return combine_match_summaries(summaries).to_dict() if summaries else {}


def _evaluate_split(
    model,
    dataset: CocoDetectionDataset,
    device,
    *,
    model_name: str,
    transformer_image_size: int,
    split: str,
    thresholds: list[float],
    iou_threshold: float,
    output_dir: Path,
    max_failure_examples: int,
) -> dict[str, Any]:
    import torch

    loader = _build_loader(dataset)
    pairs: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for index, (images, targets) in enumerate(loader):
            if is_transformer_detector(model_name):
                prediction = predict_transformer_batch(
                    model,
                    images,
                    device,
                    image_size=transformer_image_size,
                )[0]
            else:
                image_tensor = images[0].to(device)
                prediction = tensor_prediction_to_python(model([image_tensor])[0])
            target = tensor_target_to_python(targets[0])
            image_info = dataset.images[index]
            pairs.append(
                {
                    "image_info": image_info,
                    "image_path": resolve_repo_path(
                        dataset.repo_root,
                        str(image_info["file_name"]),
                    ),
                    "prediction": prediction,
                    "target": target,
                }
            )

    threshold_table = {
        f"{threshold:.2f}": _summarize_pairs(
            pairs,
            score_threshold=threshold,
            iou_threshold=iou_threshold,
        )
        for threshold in thresholds
    }

    failure_examples = []
    primary_threshold = thresholds[0]
    for pair in pairs:
        if len(failure_examples) >= max_failure_examples:
            break
        summary = match_prediction_to_target(
            pair["prediction"],
            pair["target"],
            score_threshold=primary_threshold,
            iou_threshold=iou_threshold,
        )
        has_gt = bool(pair["target"]["boxes"])
        has_pred = bool(
            [score for score in pair["prediction"]["scores"] if float(score) >= primary_threshold]
        )
        failure_type = None
        if has_gt and summary.matched_boxes < summary.ground_truth_boxes:
            failure_type = "false_negative"
        elif not has_gt and has_pred:
            failure_type = "false_positive"
        if failure_type is None:
            continue

        output_path = output_dir / "failure_examples" / split / f"{len(failure_examples):03d}.jpg"
        _draw_eval_example(
            pair["image_path"],
            output_path,
            pair["prediction"],
            pair["target"],
            threshold=primary_threshold,
        )
        failure_examples.append(
            {
                "type": failure_type,
                "image": str(pair["image_path"]),
                "visualization": str(output_path),
            }
        )

    return {
        "split": split,
        "images": len(dataset),
        "threshold_table": threshold_table,
        "failure_examples": failure_examples,
    }


def run_evaluate(config: PipelineConfig) -> None:
    """Evaluate a trained detector checkpoint on val and neg_test splits."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Evaluation requires torch and torchvision. Run `uv sync` first."
        ) from exc

    training = config.section("training")
    evaluation = config.section("evaluation")
    output_dir = resolve_repo_path(
        config.repo_root,
        evaluation.get("output_dir", "artifacts/reports/evaluation"),
    )
    checkpoint_path = resolve_repo_path(
        config.repo_root,
        evaluation.get("checkpoint", "artifacts/models/detector/detector.pt"),
    )
    thresholds = [
        float(value)
        for value in evaluation.get(
            "thresholds",
            training.get("threshold_sweep", [training.get("score_threshold", 0.5)]),
        )
    ]
    if not thresholds:
        thresholds = [float(training.get("score_threshold", 0.5))]
    iou_threshold = float(evaluation.get("iou_threshold", training.get("iou_threshold", 0.5)))
    max_failure_examples = int(evaluation.get("max_failure_examples", 12))
    transformer_image_size = int(training.get("transformer_image_size", 640))

    device = select_device(str(training.get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = str(
        checkpoint.get(
            "model_name",
            training.get("detector_model", "vit"),
        )
    )
    model = build_detection_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    split_paths = {
        "val": resolve_repo_path(
            config.repo_root,
            training.get("val_annotations", "artifacts/dataset/coco_val.json"),
        ),
        "neg_test": resolve_repo_path(
            config.repo_root,
            training.get("neg_test_annotations", "artifacts/dataset/coco_neg_test.json"),
        ),
    }
    results = {}
    for split, coco_path in split_paths.items():
        if not coco_path.exists():
            results[split] = {"split": split, "error": f"Missing COCO file: {coco_path}"}
            continue
        dataset = CocoDetectionDataset(coco_path, config.repo_root)
        results[split] = _evaluate_split(
            model,
            dataset,
            device,
            model_name=model_name,
            transformer_image_size=transformer_image_size,
            split=split,
            thresholds=thresholds,
            iou_threshold=iou_threshold,
            output_dir=output_dir,
            max_failure_examples=max_failure_examples,
        )

    report_path = output_dir / "detector_evaluation_report.json"
    _write_json(
        report_path,
        {
            "checkpoint": str(checkpoint_path),
            "model_name": model_name,
            "thresholds": thresholds,
            "iou_threshold": iou_threshold,
            "splits": results,
        },
    )
    print("Detector evaluation complete")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  report={report_path}")
