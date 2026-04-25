"""Dataset conversion and packaging entrypoints."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from prob_ml.config import PipelineConfig
from prob_ml.manifest import KitchenPhotoRecord, load_kitchen_photo_manifest
from prob_ml.render import resolve_batch_render_paths

COCO_CATEGORIES = [
    {"id": 1, "name": "mouse", "supercategory": "pest"},
    {"id": 2, "name": "rat", "supercategory": "pest"},
    {"id": 3, "name": "cockroach", "supercategory": "pest"},
]
LABEL_TO_COCO_ID = {item["name"]: int(item["id"]) for item in COCO_CATEGORIES}
LABEL_TO_YOLO_ID = {label: coco_id - 1 for label, coco_id in LABEL_TO_COCO_ID.items()}
NEGATIVE_ONLY_SPLITS = {"neg_test"}


@dataclass
class SplitArtifacts:
    """In-memory export bundle for a single positive split."""

    split: str
    coco: dict[str, object]
    image_sources: dict[int, Path]
    missing_backgrounds: list[str]


def _resolve_path(config: PipelineConfig, relative_path: str) -> Path:
    return (config.repo_root / relative_path).resolve()


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to package rendered datasets.") from exc

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _load_annotations(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list of frame annotations in {path}")
    return payload


def _base_coco(description: str) -> dict[str, object]:
    return {
        "info": {
            "description": description,
            "coco_category_ids": {label: coco_id for label, coco_id in LABEL_TO_COCO_ID.items()},
            "yolo_category_ids": {label: yolo_id for label, yolo_id in LABEL_TO_YOLO_ID.items()},
        },
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES,
    }


def _validate_bbox(
    bbox: dict[str, object],
    *,
    image_width: int,
    image_height: int,
    source_path: Path,
) -> tuple[float, float, float, float]:
    required = ["x_min", "y_min", "width", "height"]
    missing = [key for key in required if key not in bbox]
    if missing:
        raise ValueError(f"Missing bbox keys {missing} in {source_path}")

    x = float(bbox["x_min"])
    y = float(bbox["y_min"])
    width = float(bbox["width"])
    height = float(bbox["height"])

    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive bbox in {source_path}: {bbox}")
    if x < 0 or y < 0:
        raise ValueError(f"Negative bbox origin in {source_path}: {bbox}")
    if x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
        raise ValueError(f"Bbox exceeds image bounds in {source_path}: {bbox}")

    return x, y, width, height


def _group_records_by_split(
    records: list[KitchenPhotoRecord],
) -> dict[str, list[KitchenPhotoRecord]]:
    grouped: dict[str, list[KitchenPhotoRecord]] = defaultdict(list)
    for record in records:
        grouped[record.split].append(record)
    return grouped


def _positive_splits(records_by_split: dict[str, list[KitchenPhotoRecord]]) -> list[str]:
    preferred = ["train", "val", "test"]
    discovered = [
        split
        for split in records_by_split
        if split not in NEGATIVE_ONLY_SPLITS and split not in {"unused", "unassigned"}
    ]
    ordered = [split for split in preferred if split in discovered]
    ordered.extend(sorted(split for split in discovered if split not in preferred))
    return ordered


def _has_render_outputs(config: PipelineConfig, record: KitchenPhotoRecord) -> bool:
    resolved_paths = resolve_batch_render_paths(config, record)
    return resolved_paths["annotations"].exists()


def _training_val_split(config: PipelineConfig) -> float:
    raw = getattr(config, "raw", {})
    training = raw.get("training", {}) if isinstance(raw, dict) else {}
    return float(training.get("val_split", 0.2))


def _auto_assign_splits(
    config: PipelineConfig,
    records_by_split: dict[str, list[KitchenPhotoRecord]],
) -> dict[str, list[KitchenPhotoRecord]]:
    """Derive train/val/neg_test when the manifest only contains unassigned rows."""
    if "train" in records_by_split and "val" in records_by_split:
        return records_by_split

    unassigned = list(records_by_split.get("unassigned", []))
    if not unassigned:
        return records_by_split

    rendered_records = [record for record in unassigned if _has_render_outputs(config, record)]
    if len(rendered_records) < 2:
        return records_by_split

    val_fraction = _training_val_split(config)
    val_count = max(1, round(len(rendered_records) * val_fraction))
    val_count = min(val_count, len(rendered_records) - 1)
    train_records = rendered_records[:-val_count]
    val_records = rendered_records[-val_count:]
    rendered_ids = {record.image_id for record in rendered_records}
    neg_records = list(records_by_split.get("neg_test", []))
    if not neg_records:
        neg_records = [record for record in unassigned if record.image_id not in rendered_ids]

    derived = dict(records_by_split)
    derived["train"] = train_records
    derived["val"] = val_records
    if neg_records:
        derived["neg_test"] = neg_records
    return derived


def _build_positive_split(
    config: PipelineConfig,
    split: str,
    records: list[KitchenPhotoRecord],
    *,
    start_image_id: int,
    start_annotation_id: int,
) -> tuple[SplitArtifacts, int, int]:
    repo_root = config.repo_root
    coco = _base_coco(f"Synthetic pest detection split: {split}")
    image_sources: dict[int, Path] = {}
    missing_backgrounds: list[str] = []
    next_image_id = start_image_id
    next_annotation_id = start_annotation_id

    for record in records:
        resolved_paths = resolve_batch_render_paths(config, record)
        annotations_path = resolved_paths["annotations"]
        if not annotations_path.exists():
            missing_backgrounds.append(record.image_id)
            continue

        for frame in _load_annotations(annotations_path):
            frame_path = Path(str(frame["file"])).resolve()
            if not frame_path.exists():
                frame_path = resolved_paths["frames_dir"] / Path(str(frame["file"])).name
            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Annotated frame does not exist for {record.image_id}: {frame_path}"
                )

            image_width, image_height = _load_image_size(frame_path)
            coco["images"].append(
                {
                    "id": next_image_id,
                    "file_name": _to_repo_relative(frame_path, repo_root),
                    "width": image_width,
                    "height": image_height,
                    "frame_index": int(frame["frame"]),
                    "background_image_id": record.image_id,
                    "background_split": split,
                }
            )
            image_sources[next_image_id] = frame_path

            pests = frame.get("pests", [])
            if not isinstance(pests, list):
                raise TypeError(f"Frame pests must be a list in {annotations_path}")
            for pest in pests:
                label = str(pest["label"])
                if label not in LABEL_TO_COCO_ID:
                    raise ValueError(f"Unknown pest label '{label}' in {annotations_path}")
                bbox = pest["bbox"]
                if not isinstance(bbox, dict):
                    raise TypeError(f"Expected bbox dict in {annotations_path}")
                x, y, width, height = _validate_bbox(
                    bbox,
                    image_width=image_width,
                    image_height=image_height,
                    source_path=annotations_path,
                )
                coco["annotations"].append(
                    {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": LABEL_TO_COCO_ID[label],
                        "bbox": [x, y, width, height],
                        "area": round(width * height, 4),
                        "iscrowd": 0,
                    }
                )
                next_annotation_id += 1

            next_image_id += 1

    return (
        SplitArtifacts(
            split=split,
            coco=coco,
            image_sources=image_sources,
            missing_backgrounds=missing_backgrounds,
        ),
        next_image_id,
        next_annotation_id,
    )


def _build_negative_only_manifest(
    records: list[KitchenPhotoRecord],
    repo_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, Path]]:
    manifest_images: list[dict[str, object]] = []
    coco = _base_coco("Negative-only real kitchen holdout")
    image_sources: dict[str, Path] = {}
    next_image_id = 1

    for record in records:
        photo_path = record.photo_path.resolve()
        if not photo_path.exists():
            raise FileNotFoundError(f"Negative holdout image not found: {photo_path}")
        image_width, image_height = _load_image_size(photo_path)
        relative_path = _to_repo_relative(photo_path, repo_root)
        manifest_images.append(
            {
                "image_id": record.image_id,
                "file_name": relative_path,
                "width": image_width,
                "height": image_height,
            }
        )
        coco["images"].append(
            {
                "id": next_image_id,
                "file_name": relative_path,
                "width": image_width,
                "height": image_height,
                "background_image_id": record.image_id,
                "background_split": record.split,
            }
        )
        image_sources[record.image_id] = photo_path
        next_image_id += 1

    manifest = {"split": "neg_test", "images": manifest_images}
    return manifest, coco, image_sources


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        relative_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, target)


def _write_yolo_label_file(
    label_path: Path,
    image_info: dict[str, object],
    annotations: list[dict[str, object]],
) -> None:
    image_width = float(image_info["width"])
    image_height = float(image_info["height"])
    lines: list[str] = []
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        center_x = (float(x) + float(width) / 2.0) / image_width
        center_y = (float(y) + float(height) / 2.0) / image_height
        norm_width = float(width) / image_width
        norm_height = float(height) / image_height
        if not all(0.0 <= value <= 1.0 for value in [center_x, center_y, norm_width, norm_height]):
            raise ValueError(f"Invalid YOLO-normalized bbox for {label_path}")
        lines.append(
            f"{int(annotation['category_id']) - 1} "
            f"{center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
        )

    content = "\n".join(lines)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


def _export_positive_yolo(
    yolo_root: Path,
    split_artifacts: dict[str, SplitArtifacts],
) -> dict[str, str]:
    split_paths: dict[str, str] = {}
    for split, artifacts in split_artifacts.items():
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        annotations_by_image_id: dict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in artifacts.coco["annotations"]:
            annotations_by_image_id[int(annotation["image_id"])].append(annotation)

        for image in artifacts.coco["images"]:
            image_id = int(image["id"])
            source_path = artifacts.image_sources[image_id]
            basename = f"{image['background_image_id']}_frame_{int(image['frame_index']):05d}"
            image_target = images_dir / f"{basename}{source_path.suffix.lower()}"
            label_target = labels_dir / f"{basename}.txt"
            _link_or_copy(source_path, image_target)
            _write_yolo_label_file(label_target, image, annotations_by_image_id[image_id])

        split_paths[split] = f"images/{split}"
    return split_paths


def _export_negative_yolo(
    yolo_root: Path,
    manifest: dict[str, object],
    image_sources: dict[str, Path],
) -> str | None:
    images = manifest["images"]
    if not images:
        return None

    split = "neg_test"
    images_dir = yolo_root / "images" / split
    labels_dir = yolo_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        image_id = str(image["image_id"])
        source_path = image_sources[image_id]
        basename = Path(str(image["file_name"])).stem
        image_target = images_dir / f"{basename}{source_path.suffix.lower()}"
        label_target = labels_dir / f"{basename}.txt"
        _link_or_copy(source_path, image_target)
        label_target.write_text("", encoding="utf-8")

    return f"images/{split}"


def _write_yolo_dataset(
    output_root: Path,
    split_artifacts: dict[str, SplitArtifacts],
    negative_manifest: dict[str, object],
    negative_sources: dict[str, Path],
) -> Path:
    yolo_root = output_root / "yolo"
    _reset_directory(yolo_root)

    split_paths = _export_positive_yolo(yolo_root, split_artifacts)
    neg_split_path = _export_negative_yolo(yolo_root, negative_manifest, negative_sources)
    yaml_lines = [
        f"path: {yolo_root.resolve()}",
        f"train: {split_paths.get('train', '')}",
        f"val: {split_paths.get('val', split_paths.get('train', ''))}",
    ]
    if "test" in split_paths:
        yaml_lines.append(f"test: {split_paths['test']}")
    elif neg_split_path:
        yaml_lines.append(f"test: {neg_split_path}")
    yaml_lines.extend(
        [
            "names:",
            "  0: mouse",
            "  1: rat",
            "  2: cockroach",
        ]
    )
    yaml_path = yolo_root / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def convert_batch_render_outputs(config: PipelineConfig) -> dict[str, Path]:
    """Aggregate batch-render outputs into COCO and YOLO datasets."""
    inputs = config.section("inputs")
    dataset = config.section("dataset")
    manifest_path = _resolve_path(config, inputs["kitchen_manifest"])
    output_root = _resolve_path(config, dataset["coco_annotations"]).parent
    records = load_kitchen_photo_manifest(manifest_path, config.repo_root, enabled_only=False)
    records_by_split = _auto_assign_splits(config, _group_records_by_split(records))

    next_image_id = 1
    next_annotation_id = 1
    positive_artifacts: dict[str, SplitArtifacts] = {}
    for split in _positive_splits(records_by_split):
        artifacts, next_image_id, next_annotation_id = _build_positive_split(
            config,
            split,
            records_by_split[split],
            start_image_id=next_image_id,
            start_annotation_id=next_annotation_id,
        )
        if artifacts.coco["images"]:
            positive_artifacts[split] = artifacts

    if "train" not in positive_artifacts or "val" not in positive_artifacts:
        raise RuntimeError(
            "Dataset conversion requires rendered train and val splits. "
            "Check the manifest and batch-render outputs first."
        )

    negative_records = records_by_split.get("neg_test", [])
    negative_manifest, negative_coco, negative_sources = _build_negative_only_manifest(
        negative_records,
        config.repo_root,
    )

    combined_coco = _base_coco("Combined rendered pest detection dataset")
    combined_coco["images"] = [
        image
        for artifacts in positive_artifacts.values()
        for image in artifacts.coco["images"]
    ]
    combined_coco["annotations"] = [
        annotation
        for artifacts in positive_artifacts.values()
        for annotation in artifacts.coco["annotations"]
    ]

    outputs: dict[str, Path] = {}
    combined_output = _resolve_path(config, dataset["coco_annotations"])
    _write_json(combined_output, combined_coco)
    outputs["combined_coco"] = combined_output

    for split, artifacts in positive_artifacts.items():
        split_path = output_root / f"coco_{split}.json"
        _write_json(split_path, artifacts.coco)
        outputs[f"coco_{split}"] = split_path

    neg_manifest_path = output_root / "neg_test_images.json"
    neg_coco_path = output_root / "coco_neg_test.json"
    _write_json(neg_manifest_path, negative_manifest)
    _write_json(neg_coco_path, negative_coco)
    outputs["neg_test_manifest"] = neg_manifest_path
    outputs["coco_neg_test"] = neg_coco_path

    yolo_yaml = _write_yolo_dataset(
        output_root,
        positive_artifacts,
        negative_manifest,
        negative_sources,
    )
    outputs["yolo_data"] = yolo_yaml

    summary = {
        "manifest": _to_repo_relative(manifest_path, config.repo_root),
        "coco_category_ids": {label: coco_id for label, coco_id in LABEL_TO_COCO_ID.items()},
        "yolo_category_ids": {label: yolo_id for label, yolo_id in LABEL_TO_YOLO_ID.items()},
        "splits": {},
    }
    for split, artifacts in positive_artifacts.items():
        summary["splits"][split] = {
            "backgrounds_with_renders": len(
                {
                    str(image["background_image_id"])
                    for image in artifacts.coco["images"]
                }
            ),
            "frames": len(artifacts.coco["images"]),
            "annotations": len(artifacts.coco["annotations"]),
            "missing_backgrounds": artifacts.missing_backgrounds,
        }
    summary["splits"]["neg_test"] = {
        "backgrounds": len(negative_manifest["images"]),
        "annotations": 0,
    }
    summary_path = output_root / "dataset_summary.json"
    _write_json(summary_path, summary)
    outputs["summary"] = summary_path

    return outputs


def run_convert(config: PipelineConfig) -> None:
    """Convert render outputs into model-ready dataset artifacts."""
    outputs = convert_batch_render_outputs(config)
    print("Dataset conversion complete")
    for name, path in outputs.items():
        print(f"  {name}={path}")
