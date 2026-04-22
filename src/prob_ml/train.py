"""Training stage entrypoints."""

from __future__ import annotations

from prob_ml.config import PipelineConfig


def run_train(config: PipelineConfig) -> None:
    """Placeholder training entrypoint."""
    training = config.section("training")
    inputs = config.section("inputs")
    print("Training scaffold")
    print(f"  model_name={training.get('model_name')}")
    print(f"  epochs={training.get('epochs')} batch_size={training.get('batch_size')}")
    print(f"  output_dir={training.get('output_dir')}")
    print(f"  kitchen_photo_dir={inputs.get('kitchen_photo_dir')}")
    print(f"  kitchen_manifest={inputs.get('kitchen_manifest')}")
    print("  next: batch render synthetic videos from the external kitchen photo set")
