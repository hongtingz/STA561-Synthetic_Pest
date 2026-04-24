"""Dataset sanity-check and annotation visualization utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from prob_ml.config import PipelineConfig
from prob_ml.detector import CATEGORY_ID_TO_NAME, load_coco_detection_json, resolve_repo_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _coco_split_paths(config: PipelineConfig) -> dict[str, Path]:
    training = config.section("training")
    return {
        "train": resolve_repo_path(
            config.repo_root,
            training.get("train_annotations", "artifacts/dataset/coco_train.json"),
        ),
        "val": resolve_repo_path(
            config.repo_root,
            training.get("val_annotations", "artifacts/dataset/coco_val.json"),
        ),
        "neg_test": resolve_repo_path(
            config.repo_root,
            training.get("neg_test_annotations", "artifacts/dataset/coco_neg_test.json"),
        ),
    }


def _background_key(image: dict[str, Any]) -> str:
    if image.get("background_image_id") is not None:
        return str(image["background_image_id"])
    return Path(str(image["file_name"])).stem


def _draw_annotation_overlay(
    image_path: Path,
    output_path: Path,
    image_info: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except OSError:
            font = None

        for annotation in annotations:
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            label = CATEGORY_ID_TO_NAME.get(
                int(annotation["category_id"]),
                str(annotation["category_id"]),
            )
            draw.rectangle([x, y, x + width, y + height], outline=(40, 180, 90), width=3)
            text_box = draw.textbbox((x, y), label, font=font)
            draw.rectangle(text_box, fill=(40, 180, 90))
            draw.text((x, y), label, fill=(255, 255, 255), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def inspect_coco_file(
    coco_path: Path,
    repo_root: Path,
    *,
    split: str,
    overlay_dir: Path | None = None,
    max_overlay_images: int = 0,
) -> dict[str, Any]:
    """Validate one COCO file and optionally write bbox overlay images."""
    payload = load_coco_detection_json(coco_path)
    images = payload["images"]
    annotations = payload["annotations"]
    images_by_id = {int(image["id"]): image for image in images}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in images_by_id
    }
    for annotation in annotations:
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[str] = Counter()
    background_keys = {_background_key(image) for image in images}

    for image in images:
        image_path = resolve_repo_path(repo_root, str(image["file_name"]))
        if not image_path.exists():
            errors.append(f"{split}: missing image file {image_path}")

    for annotation in annotations:
        image_id = int(annotation["image_id"])
        image = images_by_id.get(image_id)
        if image is None:
            errors.append(f"{split}: annotation {annotation.get('id')} references {image_id}")
            continue
        category_id = int(annotation["category_id"])
        class_counts[CATEGORY_ID_TO_NAME.get(category_id, str(category_id))] += 1
        if category_id not in CATEGORY_ID_TO_NAME:
            errors.append(f"{split}: unknown category id {category_id}")
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        if width <= 0 or height <= 0:
            errors.append(f"{split}: non-positive bbox in annotation {annotation.get('id')}")
        if x < 0 or y < 0:
            errors.append(f"{split}: negative bbox origin in annotation {annotation.get('id')}")
        if x + width > float(image["width"]) + 1e-6:
            errors.append(f"{split}: bbox exceeds width in annotation {annotation.get('id')}")
        if y + height > float(image["height"]) + 1e-6:
            errors.append(f"{split}: bbox exceeds height in annotation {annotation.get('id')}")

    if split != "neg_test" and not annotations:
        warnings.append(f"{split}: positive split has no annotations")
    if split == "neg_test" and annotations:
        warnings.append("neg_test: expected zero annotations for negative-only split")

    overlay_outputs: list[str] = []
    if overlay_dir is not None and max_overlay_images > 0:
        drawn = 0
        for image in images:
            if drawn >= max_overlay_images:
                break
            image_annotations = annotations_by_image_id.get(int(image["id"]), [])
            if split != "neg_test" and not image_annotations:
                continue
            image_path = resolve_repo_path(repo_root, str(image["file_name"]))
            if not image_path.exists():
                continue
            output_path = overlay_dir / split / f"{Path(str(image['file_name'])).stem}.jpg"
            _draw_annotation_overlay(image_path, output_path, image, image_annotations)
            overlay_outputs.append(str(output_path))
            drawn += 1

    return {
        "split": split,
        "coco_path": str(coco_path),
        "images": len(images),
        "annotations": len(annotations),
        "class_counts": dict(class_counts),
        "background_keys": sorted(background_keys),
        "errors": errors,
        "warnings": warnings,
        "overlay_outputs": overlay_outputs,
    }


def run_sanity_check(config: PipelineConfig) -> None:
    """Run COCO split checks and write a dataset QA report."""
    sanity = config.section("sanity")
    output_dir = resolve_repo_path(config.repo_root, sanity.get("output_dir", "artifacts/reports"))
    overlay_dir = output_dir / "sanity_overlays"
    max_overlay_images = int(sanity.get("max_overlay_images", 12))

    reports = {}
    for split, coco_path in _coco_split_paths(config).items():
        if not coco_path.exists():
            reports[split] = {
                "split": split,
                "coco_path": str(coco_path),
                "images": 0,
                "annotations": 0,
                "class_counts": {},
                "background_keys": [],
                "errors": [f"Missing COCO file: {coco_path}"],
                "warnings": [],
                "overlay_outputs": [],
            }
            continue
        reports[split] = inspect_coco_file(
            coco_path,
            config.repo_root,
            split=split,
            overlay_dir=overlay_dir,
            max_overlay_images=max_overlay_images,
        )

    leakage = []
    split_names = list(reports)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(
                set(reports[left]["background_keys"]) & set(reports[right]["background_keys"])
            )
            if overlap:
                leakage.append({"splits": [left, right], "background_keys": overlap})

    all_errors = [error for report in reports.values() for error in report["errors"]]
    all_warnings = [warning for report in reports.values() for warning in report["warnings"]]
    if leakage:
        all_errors.append("Background leakage detected across splits")

    payload = {
        "status": "pass" if not all_errors else "fail",
        "splits": reports,
        "background_leakage": leakage,
        "errors": all_errors,
        "warnings": all_warnings,
    }
    report_path = output_dir / "dataset_sanity_report.json"
    _write_json(report_path, payload)

    print("Dataset sanity check complete")
    print(f"  status={payload['status']}")
    print(f"  report={report_path}")
    print(f"  errors={len(all_errors)}")
    print(f"  warnings={len(all_warnings)}")
