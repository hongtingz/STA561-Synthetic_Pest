"""Inference and evaluation entrypoints."""

from __future__ import annotations

from prob_ml.config import PipelineConfig


def run_infer(config: PipelineConfig) -> None:
    """Placeholder inference entrypoint."""
    inference = config.section("inference")
    print("Inference scaffold")
    print(f"  input_image={inference.get('input_image')}")
    print(f"  output_image={inference.get('output_image')}")
    print(f"  threshold={inference.get('threshold')}")
    print("  next: implement detector inference and evaluation reports")

