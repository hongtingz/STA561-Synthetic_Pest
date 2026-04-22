"""Configuration loading for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Union

from prob_ml.paths import find_repo_root


@dataclass
class PipelineConfig:
    """Resolved project configuration."""

    path: Path
    raw: dict
    repo_root: Path

    def section(self, name: str) -> dict:
        """Return a config section or an empty mapping."""
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section '{name}' must be a table.")
        return value


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load a JSON config file and attach the resolved repository root."""
    repo_root = find_repo_root()
    path = Path(config_path)
    resolved_path = path if path.is_absolute() else (repo_root / path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return PipelineConfig(path=resolved_path, raw=raw, repo_root=repo_root)
