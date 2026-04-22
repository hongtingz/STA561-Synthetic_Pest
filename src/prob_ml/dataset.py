"""Dataset conversion and packaging entrypoints."""

from __future__ import annotations

from prob_ml.config import PipelineConfig


def run_convert(config: PipelineConfig) -> None:
    """Placeholder dataset conversion entrypoint."""
    dataset = config.section("dataset")
    print("Dataset conversion scaffold")
    print(f"  raw_annotations={dataset.get('annotations_raw')}")
    print(f"  coco_annotations={dataset.get('coco_annotations')}")
    print("  next: implement COCO/video export and split generation")

