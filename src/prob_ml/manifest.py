"""Manifest helpers for batch kitchen-photo processing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KitchenPhotoRecord:
    """Single kitchen-photo entry resolved from the manifest."""

    image_id: str
    filename: str
    photo_path: Path
    split: str
    enabled: bool


def _parse_enabled(raw_value: str | None) -> bool:
    value = (raw_value or "true").strip().lower()
    return value not in {"0", "false", "no", "n", "off"}


def load_kitchen_photo_manifest(
    manifest_path: Path,
    repo_root: Path,
    *,
    limit: int | None = None,
    enabled_only: bool = True,
) -> list[KitchenPhotoRecord]:
    """Load kitchen-photo records from a CSV manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Kitchen manifest not found: {manifest_path}")

    records: list[KitchenPhotoRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            relative_path = (row.get("relative_path") or "").strip()
            filename = (row.get("filename") or "").strip()
            image_id = (row.get("image_id") or "").strip()

            if not relative_path and not filename:
                raise ValueError(
                    f"Manifest row {row_index} must define relative_path or filename."
                )

            path = Path(relative_path) if relative_path else Path(filename)
            if not path.is_absolute():
                path = (repo_root / path).resolve()

            if not image_id:
                image_id = path.stem

            enabled = _parse_enabled(row.get("enabled"))
            if enabled_only and not enabled:
                continue

            if not path.exists():
                raise FileNotFoundError(
                    f"Manifest image not found for row {row_index}: {path}"
                )

            records.append(
                KitchenPhotoRecord(
                    image_id=image_id,
                    filename=filename or path.name,
                    photo_path=path,
                    split=(row.get("split") or "unassigned").strip() or "unassigned",
                    enabled=enabled,
                )
            )

            if limit is not None and len(records) >= limit:
                break

    return records
