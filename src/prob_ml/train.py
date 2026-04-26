"""Training stage entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from prob_ml.config import PipelineConfig
from prob_ml.detector import (
    CocoDetectionDataset,
    build_detection_model,
    checkpoint_payload,
    collate_detection_batch,
    combine_match_summaries,
    is_transformer_detector,
    match_prediction_to_target,
    move_targets_to_device,
    prepare_transformer_training_batch,
    predict_transformer_batch,
    resolve_repo_path,
    select_device,
    tensor_prediction_to_python,
    tensor_target_to_python,
)


def _resolve_path(config: PipelineConfig, raw_path: str | Path) -> Path:
    return resolve_repo_path(config.repo_root, raw_path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    torch_module,
    model,
    model_name: str,
    metrics: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        checkpoint_payload(model, model_name=model_name, metrics=metrics),
        checkpoint_path,
    )


def _build_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
    )


def _collect_prediction_pairs(
    model,
    loader,
    device,
    *,
    model_name: str,
    transformer_image_size: int,
):
    import torch

    pairs = []
    model.eval()
    for images, targets in loader:
        with torch.no_grad():
            if is_transformer_detector(model_name):
                predictions = predict_transformer_batch(
                    model,
                    images,
                    device,
                    image_size=transformer_image_size,
                )
            else:
                image_tensors = [image.to(device) for image in images]
                predictions = [
                    tensor_prediction_to_python(prediction)
                    for prediction in model(image_tensors)
                ]
        for prediction, target in zip(predictions, targets, strict=True):
            pairs.append(
                (
                    prediction,
                    tensor_target_to_python(target),
                )
            )
    return pairs


def _summarize_prediction_pairs(
    pairs,
    *,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    summaries = [
        match_prediction_to_target(
            prediction,
            target,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        for prediction, target in pairs
    ]
    return combine_match_summaries(summaries).to_dict() if summaries else {}


def _evaluate_loader(
    model,
    loader,
    device,
    *,
    model_name: str,
    transformer_image_size: int,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    return _summarize_prediction_pairs(
        _collect_prediction_pairs(
            model,
            loader,
            device,
            model_name=model_name,
            transformer_image_size=transformer_image_size,
        ),
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )


def _evaluate_threshold_sweep(
    model,
    loader,
    device,
    *,
    model_name: str,
    transformer_image_size: int,
    thresholds: list[float],
    iou_threshold: float,
) -> dict[str, Any]:
    pairs = _collect_prediction_pairs(
        model,
        loader,
        device,
        model_name=model_name,
        transformer_image_size=transformer_image_size,
    )
    return {
        f"{threshold:.2f}": _summarize_prediction_pairs(
            pairs,
            score_threshold=threshold,
            iou_threshold=iou_threshold,
        )
        for threshold in thresholds
    }


def run_train(config: PipelineConfig) -> None:
    """Train the configured detector for the current pipeline."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training requires torch and torchvision. Run `uv sync` in the project "
            "environment before `pest-pipeline train`."
        ) from exc

    training = config.section("training")
    dataset = config.section("dataset")
    output_dir = _resolve_path(config, training.get("output_dir", "artifacts/models/detector"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_annotations = _resolve_path(
        config,
        training.get("train_annotations", "artifacts/dataset/coco_train.json"),
    )
    val_annotations = _resolve_path(
        config,
        training.get("val_annotations", "artifacts/dataset/coco_val.json"),
    )
    neg_annotations = _resolve_path(
        config,
        training.get("neg_test_annotations", "artifacts/dataset/coco_neg_test.json"),
    )
    model_name = str(
        training.get("detector_model", "vit")
    )
    epochs = int(training.get("epochs", 5))
    batch_size = int(training.get("batch_size", 2))
    learning_rate = float(training.get("learning_rate", 1e-4))
    weight_decay = float(training.get("weight_decay", 1e-4))
    num_workers = int(training.get("num_workers", 0))
    score_threshold = float(training.get("score_threshold", 0.5))
    iou_threshold = float(training.get("iou_threshold", 0.5))
    threshold_sweep = [float(value) for value in training.get("threshold_sweep", [])]
    pretrained = bool(training.get("pretrained", False))
    checkpoint_interval = max(1, int(training.get("checkpoint_interval", 3)))
    augmentation = training.get("augmentation", {})
    if augmentation is None:
        augmentation = {}
    if not isinstance(augmentation, dict):
        raise TypeError("training.augmentation must be a JSON object.")
    max_train_images = training.get("max_train_images")
    max_val_images = training.get("max_val_images")
    transformer_image_size = int(training.get("transformer_image_size", 640))

    print("Detector training")
    print(f"  model={model_name}")
    print(f"  train_annotations={train_annotations}")
    print(f"  val_annotations={val_annotations}")
    print(f"  neg_test_annotations={neg_annotations}")
    print(f"  pretrained={pretrained}")
    print(f"  checkpoint_interval={checkpoint_interval}")
    print(f"  transformer_image_size={transformer_image_size}")
    print(f"  threshold_sweep={threshold_sweep or [score_threshold]}")
    print(f"  coco_annotations={dataset.get('coco_annotations')}")
    print(f"  output_dir={output_dir}")

    train_dataset = CocoDetectionDataset(
        train_annotations,
        config.repo_root,
        max_images=int(max_train_images) if max_train_images else None,
        augment=augmentation,
    )
    val_dataset = CocoDetectionDataset(
        val_annotations,
        config.repo_root,
        max_images=int(max_val_images) if max_val_images else None,
    )
    neg_dataset = (
        CocoDetectionDataset(neg_annotations, config.repo_root)
        if neg_annotations.exists()
        else None
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Training requires non-empty train and val COCO exports.")

    train_loader = _build_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    neg_loader = (
        _build_loader(neg_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        if neg_dataset is not None and len(neg_dataset) > 0
        else None
    )

    device = select_device(str(training.get("device", "auto")))
    model = build_detection_model(model_name, pretrained=pretrained)
    model.to(device)
    checkpoints_dir = output_dir / "checkpoints"
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    saved_checkpoints: list[str] = []
    for epoch in range(1, epochs + 1):
        start_time = perf_counter()
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, targets in train_loader:
            if is_transformer_detector(model_name):
                pixel_values, labels = prepare_transformer_training_batch(
                    images,
                    targets,
                    device,
                    image_size=transformer_image_size,
                )
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            else:
                images = [image.to(device) for image in images]
                targets = move_targets_to_device(targets, device)
                loss_dict = model(images, targets)
                loss = sum(loss_value for loss_value in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batch_count += 1

        avg_loss = running_loss / max(1, batch_count)
        val_metrics = _evaluate_loader(
            model,
            val_loader,
            device,
            model_name=model_name,
            transformer_image_size=transformer_image_size,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "seconds": round(perf_counter() - start_time, 3),
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        print(
            f"  epoch={epoch}/{epochs} loss={avg_loss:.4f} "
            f"val_tdr={val_metrics.get('true_detection_rate', 0.0):.3f}"
        )
        if epoch % checkpoint_interval == 0:
            epoch_checkpoint = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            _save_checkpoint(
                epoch_checkpoint,
                torch_module=torch,
                model=model,
                model_name=model_name,
                metrics={
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_metrics": val_metrics,
                },
            )
            saved_checkpoints.append(str(epoch_checkpoint))
            print(f"  saved_checkpoint={epoch_checkpoint}")

    neg_metrics = (
        _evaluate_loader(
            model,
            neg_loader,
            device,
            model_name=model_name,
            transformer_image_size=transformer_image_size,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        if neg_loader is not None
        else {}
    )
    sweep_thresholds = threshold_sweep or [score_threshold]
    threshold_sweep_metrics = {
        "validation": _evaluate_threshold_sweep(
            model,
            val_loader,
            device,
            model_name=model_name,
            transformer_image_size=transformer_image_size,
            thresholds=sweep_thresholds,
            iou_threshold=iou_threshold,
        ),
        "neg_test": (
            _evaluate_threshold_sweep(
                model,
                neg_loader,
                device,
                model_name=model_name,
                transformer_image_size=transformer_image_size,
                thresholds=sweep_thresholds,
                iou_threshold=iou_threshold,
            )
            if neg_loader is not None
            else {}
        ),
    }
    final_metrics = {
        "validation": history[-1]["val_metrics"] if history else {},
        "neg_test": neg_metrics,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "threshold_sweep": threshold_sweep_metrics,
    }
    checkpoint_path = output_dir / "detector.pt"
    _save_checkpoint(
        checkpoint_path,
        torch_module=torch,
        model=model,
        model_name=model_name,
        metrics=final_metrics,
    )
    report_path = output_dir / "training_report.json"
    _write_json(
        report_path,
        {
            "model_name": model_name,
            "train_annotations": str(train_annotations),
            "val_annotations": str(val_annotations),
            "neg_test_annotations": str(neg_annotations),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pretrained": pretrained,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_dir": str(checkpoints_dir),
            "saved_checkpoints": saved_checkpoints,
            "transformer_image_size": transformer_image_size,
            "augmentation": augmentation,
            "dataset_counts": {
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
                "neg_test_images": len(neg_dataset) if neg_dataset is not None else 0,
            },
            "history": history,
            "final_metrics": final_metrics,
            "checkpoint": str(checkpoint_path),
        },
    )
    print(f"  checkpoint={checkpoint_path}")
    print(f"  training_report={report_path}")
