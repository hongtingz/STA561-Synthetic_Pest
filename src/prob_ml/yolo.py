"""Optional Ultralytics YOLO training entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prob_ml.config import PipelineConfig
from prob_ml.detector import resolve_repo_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_train_yolo(config: PipelineConfig) -> None:
    """Train an optional YOLO detector if ultralytics is installed."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optional YOLO training requires ultralytics. Install it with "
            "`uv add ultralytics` or keep using `pest-pipeline train` for the "
            "built-in Faster R-CNN baseline."
        ) from exc

    yolo = config.section("yolo")
    data_yaml = resolve_repo_path(
        config.repo_root,
        yolo.get("data_yaml", "artifacts/dataset/yolo/data.yaml"),
    )
    output_dir = resolve_repo_path(
        config.repo_root,
        yolo.get("output_dir", "artifacts/models/yolo"),
    )
    model_name = str(yolo.get("model", "yolov8n.pt"))
    epochs = int(yolo.get("epochs", 20))
    image_size = int(yolo.get("imgsz", 640))
    batch_size = int(yolo.get("batch", 8))
    workers = int(yolo.get("workers", 4))
    run_name = str(yolo.get("name", "pest-yolo"))
    device = str(yolo.get("device", "auto"))

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO data file not found: {data_yaml}. Run `pest-pipeline convert` first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print("YOLO training")
    print(f"  model={model_name}")
    print(f"  data_yaml={data_yaml}")
    print(f"  output_dir={output_dir}")
    print(f"  epochs={epochs}")
    print(f"  imgsz={image_size}")
    print(f"  batch={batch_size}")

    model = YOLO(model_name)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "workers": workers,
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": True,
    }
    if device != "auto":
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    report_path = output_dir / "yolo_training_report.json"
    _write_json(
        report_path,
        {
            "model": model_name,
            "data_yaml": str(data_yaml),
            "epochs": epochs,
            "imgsz": image_size,
            "batch": batch_size,
            "workers": workers,
            "device": device,
            "run_dir": str(save_dir) if save_dir else str(output_dir / run_name),
            "weights_hint": str(output_dir / run_name / "weights" / "best.pt"),
        },
    )
    print(f"  yolo_report={report_path}")
