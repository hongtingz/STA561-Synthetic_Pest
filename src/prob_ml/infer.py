"""Inference and evaluation entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prob_ml.config import PipelineConfig
from prob_ml.detector import (
    CATEGORY_ID_TO_NAME,
    build_detection_model,
    is_transformer_detector,
    predict_transformer_batch,
    resolve_repo_path,
    select_device,
    tensor_prediction_to_python,
)


def _resolve_path(config: PipelineConfig, raw_path: str | Path) -> Path:
    return resolve_repo_path(config.repo_root, raw_path)


def _draw_prediction(
    image_path: Path,
    output_path: Path,
    prediction: dict[str, Any],
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

        for box, label, score in zip(
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
            strict=True,
        ):
            if float(score) < threshold:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            class_name = CATEGORY_ID_TO_NAME.get(int(label), str(label))
            text = f"{class_name} {float(score):.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(240, 60, 40), width=3)
            text_box = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(text_box, fill=(240, 60, 40))
            draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def run_infer(config: PipelineConfig) -> None:
    """Run detector inference on a single image and save visualization + JSON."""
    try:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as transform_functional
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference requires torch, torchvision, and Pillow. Run `uv sync` first."
        ) from exc

    inference = config.section("inference")
    training = config.section("training")
    default_checkpoint = (
        _resolve_path(config, training.get("output_dir", "artifacts/models/detector"))
        / "detector.pt"
    )
    checkpoint_path = _resolve_path(
        config,
        inference.get("checkpoint", str(default_checkpoint)),
    )
    input_image = _resolve_path(config, inference.get("input_image", "data/inputs/kitchen.jpg"))
    output_image = _resolve_path(
        config,
        inference.get("output_image", "artifacts/infer/infer_result.jpg"),
    )
    predictions_json = _resolve_path(
        config,
        inference.get("predictions_json", "artifacts/infer/predictions.json"),
    )
    threshold = float(inference.get("threshold", 0.5))
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
    model.eval()

    with Image.open(input_image) as image:
        image_tensor = transform_functional.to_tensor(image.convert("RGB"))
    with torch.no_grad():
        if is_transformer_detector(model_name):
            prediction_payload = predict_transformer_batch(
                model,
                [image_tensor],
                device,
                image_size=transformer_image_size,
            )[0]
        else:
            prediction = model([image_tensor.to(device)])[0]
            prediction_payload = tensor_prediction_to_python(prediction)
    filtered_payload = {
        "image": str(input_image),
        "threshold": threshold,
        "detections": [
            {
                "bbox_xyxy": box,
                "category_id": int(label),
                "label": CATEGORY_ID_TO_NAME.get(int(label), str(label)),
                "score": float(score),
            }
            for box, label, score in zip(
                prediction_payload["boxes"],
                prediction_payload["labels"],
                prediction_payload["scores"],
                strict=True,
            )
            if float(score) >= threshold
        ],
    }
    predictions_json.parent.mkdir(parents=True, exist_ok=True)
    with predictions_json.open("w", encoding="utf-8") as handle:
        json.dump(filtered_payload, handle, indent=2)
    _draw_prediction(input_image, output_image, prediction_payload, threshold=threshold)

    print("Detector inference complete")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  input_image={input_image}")
    print(f"  output_image={output_image}")
    print(f"  predictions_json={predictions_json}")
    print(f"  detections={len(filtered_payload['detections'])}")
