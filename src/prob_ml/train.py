"""Training stage entrypoints."""

from __future__ import annotations

from prob_ml.config import PipelineConfig


def run_train(config: PipelineConfig) -> None:
    """Placeholder training entrypoint."""
    training = config.section("training")
    print("Training scaffold")
    print(f"  model_name={training.get('model_name')}")
    print(f"  epochs={training.get('epochs')} batch_size={training.get('batch_size')}")
    print(f"  output_dir={training.get('output_dir')}")
    print("  next: implement ViT training + evaluation on generated data")

