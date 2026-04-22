"""Tests for kitchen-photo manifest loading."""

from __future__ import annotations

from pathlib import Path

from prob_ml.manifest import load_kitchen_photo_manifest


def test_load_kitchen_photo_manifest_respects_enabled_and_limit(
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_c = tmp_path / "c.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    image_c.write_bytes(b"c")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,filename,relative_path,split,enabled",
                "kitchen_0001,a.jpg,a.jpg,train,true",
                "kitchen_0002,b.jpg,b.jpg,val,false",
                "kitchen_0003,c.jpg,c.jpg,test,true",
            ]
        ),
        encoding="utf-8",
    )

    records = load_kitchen_photo_manifest(
        manifest_path,
        tmp_path,
        limit=1,
        enabled_only=True,
    )

    assert len(records) == 1
    assert records[0].image_id == "kitchen_0001"
    assert records[0].photo_path == image_a.resolve()
    assert records[0].split == "train"


def test_load_kitchen_photo_manifest_keeps_disabled_when_requested(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "kitchen.jpg"
    image_path.write_bytes(b"image")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,filename,relative_path,split,enabled",
                "kitchen_0001,kitchen.jpg,kitchen.jpg,unassigned,false",
            ]
        ),
        encoding="utf-8",
    )

    records = load_kitchen_photo_manifest(
        manifest_path,
        tmp_path,
        enabled_only=False,
    )

    assert len(records) == 1
    assert records[0].enabled is False
