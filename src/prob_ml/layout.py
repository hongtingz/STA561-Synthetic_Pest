"""Layout-spec generation and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple


def _rounded_pair(values: Tuple[float, float]) -> List[float]:
    return [round(values[0], 3), round(values[1], 3)]


def _rounded_triplet(values: Tuple[float, float, float]) -> List[float]:
    return [round(values[0], 3), round(values[1], 3), round(values[2], 3)]


@dataclass
class RoomSpec:
    """Simple rectangular room model inferred from the kitchen photo."""

    width: float
    depth: float
    height: float


@dataclass
class CameraSpec:
    """Camera pose for the synthetic scene."""

    location: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    lens_mm: float


@dataclass
class LightSpec:
    """Basic light configuration."""

    name: str
    kind: str
    location: Tuple[float, float, float]
    energy: float
    color: Tuple[float, float, float]


@dataclass
class FixtureSpec:
    """Axis-aligned kitchen fixture."""

    name: str
    kind: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color: Tuple[float, float, float]


@dataclass
class PestPathSpec:
    """Linear motion path for a pest instance."""

    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    elevation: float


@dataclass
class PestInstanceSpec:
    """Renderable pest instance with a path."""

    pest_id: str
    pest_type: str
    scale: float
    path: PestPathSpec


@dataclass
class LayoutSpec:
    """Serializable kitchen layout spec."""

    schema_version: str
    source_photo: str
    photo_size: Tuple[int, int]
    room: RoomSpec
    camera: CameraSpec
    lights: List[LightSpec]
    fixtures: List[FixtureSpec]
    pests: List[PestInstanceSpec]

    def to_dict(self) -> Dict[str, object]:
        """Convert the layout spec to a JSON-serializable mapping."""
        raw = asdict(self)
        raw["room"] = {
            "width": round(self.room.width, 3),
            "depth": round(self.room.depth, 3),
            "height": round(self.room.height, 3),
        }
        raw["camera"] = {
            "location": _rounded_triplet(self.camera.location),
            "rotation_deg": _rounded_triplet(self.camera.rotation_deg),
            "lens_mm": round(self.camera.lens_mm, 3),
        }
        raw["lights"] = [
            {
                "name": light.name,
                "kind": light.kind,
                "location": _rounded_triplet(light.location),
                "energy": round(light.energy, 3),
                "color": _rounded_triplet(light.color),
            }
            for light in self.lights
        ]
        raw["fixtures"] = [
            {
                "name": fixture.name,
                "kind": fixture.kind,
                "center": _rounded_triplet(fixture.center),
                "size": _rounded_triplet(fixture.size),
                "color": _rounded_triplet(fixture.color),
            }
            for fixture in self.fixtures
        ]
        raw["pests"] = [
            {
                "pest_id": pest.pest_id,
                "pest_type": pest.pest_type,
                "scale": round(pest.scale, 3),
                "path": {
                    "start": _rounded_triplet(pest.path.start),
                    "end": _rounded_triplet(pest.path.end),
                    "elevation": round(pest.path.elevation, 3),
                },
            }
            for pest in self.pests
        ]
        raw["photo_size"] = [int(self.photo_size[0]), int(self.photo_size[1])]
        return raw


def save_layout_spec(layout_spec: LayoutSpec, path: Path) -> None:
    """Write a layout spec to disk as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(layout_spec.to_dict(), handle, indent=2)


def load_layout_spec(path: Path) -> Dict[str, object]:
    """Load a previously generated layout spec."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_layout_spec(
    photo_path: Path,
    photo_size: Tuple[int, int],
    pest_types: Sequence[str],
    scene_seed: int,
) -> LayoutSpec:
    """Infer a simple room layout and pest plan from an input kitchen photo."""
    width_px, height_px = photo_size
    aspect_ratio = width_px / max(height_px, 1)
    rng = random.Random(scene_seed)

    room_width = 5.8 + min(2.2, max(0.0, aspect_ratio - 1.2))
    room_depth = 5.0 + min(1.8, max(0.0, 1.9 - aspect_ratio))
    room_height = 2.8

    counter_depth = 0.65
    cabinet_height = 0.9
    wall_y = room_depth / 2.0

    fixtures = [
        FixtureSpec(
            name="counter_back",
            kind="counter",
            center=(0.0, wall_y - counter_depth / 2.0, cabinet_height),
            size=(room_width * 0.8, counter_depth, 0.08),
            color=(0.55, 0.53, 0.5),
        ),
        FixtureSpec(
            name="cabinet_left",
            kind="cabinet",
            center=(-room_width * 0.22, wall_y - counter_depth / 2.0, cabinet_height / 2.0),
            size=(1.3, counter_depth, cabinet_height),
            color=(0.42, 0.29, 0.18),
        ),
        FixtureSpec(
            name="cabinet_right",
            kind="cabinet",
            center=(room_width * 0.24, wall_y - counter_depth / 2.0, cabinet_height / 2.0),
            size=(1.8, counter_depth, cabinet_height),
            color=(0.37, 0.26, 0.16),
        ),
        FixtureSpec(
            name="fridge",
            kind="fridge",
            center=(-room_width * 0.42, wall_y - 0.42, 1.05),
            size=(0.9, 0.85, 2.1),
            color=(0.86, 0.86, 0.88),
        ),
        FixtureSpec(
            name="island_table",
            kind="table",
            center=(room_width * 0.26, -room_depth * 0.1, 0.42),
            size=(1.0, 0.7, 0.84),
            color=(0.46, 0.31, 0.18),
        ),
    ]

    pests: List[PestInstanceSpec] = []
    base_paths = [
        ((-room_width * 0.3, -room_depth * 0.15, 0.0), (room_width * 0.2, -room_depth * 0.05, 0.0)),
        ((room_width * 0.25, 0.0, 0.0), (-room_width * 0.1, room_depth * 0.08, 0.0)),
        ((-room_width * 0.1, room_depth * 0.12, 0.0), (room_width * 0.3, room_depth * 0.18, 0.0)),
    ]

    for index, pest_type in enumerate(pest_types):
        start, end = base_paths[index % len(base_paths)]
        jitter_x = rng.uniform(-0.12, 0.12)
        jitter_y = rng.uniform(-0.1, 0.1)
        scale = 1.0 if pest_type == "mouse" else 1.25 if pest_type == "rat" else 0.55
        elevation = 0.08 if pest_type in {"mouse", "rat"} else 0.03
        pests.append(
            PestInstanceSpec(
                pest_id=f"{pest_type}_{index + 1}",
                pest_type=pest_type,
                scale=scale,
                path=PestPathSpec(
                    start=(start[0] + jitter_x, start[1] + jitter_y, elevation),
                    end=(end[0] + jitter_x, end[1] + jitter_y, elevation),
                    elevation=elevation,
                ),
            )
        )

    lights = [
        LightSpec(
            name="key",
            kind="AREA",
            location=(0.0, 0.0, room_height + 1.5),
            energy=450.0,
            color=(1.0, 0.95, 0.88),
        ),
        LightSpec(
            name="fill",
            kind="AREA",
            location=(-room_width * 0.4, -room_depth * 0.35, room_height + 0.5),
            energy=180.0,
            color=(0.88, 0.92, 1.0),
        ),
        LightSpec(
            name="rim",
            kind="POINT",
            location=(room_width * 0.32, wall_y - 0.2, room_height + 0.2),
            energy=70.0,
            color=(1.0, 0.93, 0.84),
        ),
    ]

    camera = CameraSpec(
        location=(0.0, -(room_depth + 1.2), room_height + 1.8),
        rotation_deg=(58.0, 0.0, 0.0),
        lens_mm=32.0,
    )

    return LayoutSpec(
        schema_version="0.1.0",
        source_photo=str(photo_path),
        photo_size=photo_size,
        room=RoomSpec(width=room_width, depth=room_depth, height=room_height),
        camera=camera,
        lights=lights,
        fixtures=fixtures,
        pests=pests,
    )
