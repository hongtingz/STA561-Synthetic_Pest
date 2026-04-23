"""Layout-spec generation and photo-driven scene heuristics."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _rounded_triplet(values: tuple[float, float, float]) -> list[float]:
    return [round(values[0], 3), round(values[1], 3), round(values[2], 3)]


def photo_cues_to_dict(photo_cues: PhotoCuesSpec) -> dict[str, float]:
    """Convert photo cues to a stable JSON-friendly mapping."""
    return {
        "brightness_top": round(photo_cues.brightness_top, 4),
        "brightness_mid": round(photo_cues.brightness_mid, 4),
        "brightness_bottom": round(photo_cues.brightness_bottom, 4),
        "left_brightness": round(photo_cues.left_brightness, 4),
        "center_brightness": round(photo_cues.center_brightness, 4),
        "right_brightness": round(photo_cues.right_brightness, 4),
        "floor_line_ratio": round(photo_cues.floor_line_ratio, 4),
        "clutter_score": round(photo_cues.clutter_score, 4),
        "warm_bias": round(photo_cues.warm_bias, 4),
        "contrast_score": round(photo_cues.contrast_score, 4),
    }


@dataclass
class PhotoCuesSpec:
    """Lightweight visual cues extracted from the kitchen photo."""

    brightness_top: float
    brightness_mid: float
    brightness_bottom: float
    left_brightness: float
    center_brightness: float
    right_brightness: float
    floor_line_ratio: float
    clutter_score: float
    warm_bias: float
    contrast_score: float


@dataclass
class RoomSpec:
    """Simple rectangular room model inferred from the kitchen photo."""

    width: float
    depth: float
    height: float


@dataclass
class CameraSpec:
    """Camera pose for the synthetic scene."""

    location: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    lens_mm: float


@dataclass
class LightSpec:
    """Basic light configuration."""

    name: str
    kind: str
    location: tuple[float, float, float]
    energy: float
    color: tuple[float, float, float]


@dataclass
class FixtureSpec:
    """Axis-aligned kitchen fixture."""

    name: str
    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass
class PestPathSpec:
    """Motion path for a pest instance (optionally with weaving waypoints)."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    elevation: float
    waypoints: tuple[tuple[float, float, float], ...] = ()


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
    photo_size: tuple[int, int]
    photo_cues: PhotoCuesSpec
    room: RoomSpec
    camera: CameraSpec
    lights: list[LightSpec]
    fixtures: list[FixtureSpec]
    pests: list[PestInstanceSpec]

    def to_dict(self) -> dict[str, object]:
        """Convert the layout spec to a JSON-serializable mapping."""
        raw = asdict(self)
        raw["photo_size"] = [int(self.photo_size[0]), int(self.photo_size[1])]
        raw["photo_cues"] = photo_cues_to_dict(self.photo_cues)
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
        raw["pests"] = []
        for pest in self.pests:
            path_dict: dict[str, object] = {
                "start": _rounded_triplet(pest.path.start),
                "end": _rounded_triplet(pest.path.end),
                "elevation": round(pest.path.elevation, 3),
            }
            if pest.path.waypoints:
                path_dict["waypoints"] = [
                    _rounded_triplet(p) for p in pest.path.waypoints
                ]
            raw["pests"].append(
                {
                    "pest_id": pest.pest_id,
                    "pest_type": pest.pest_type,
                    "scale": round(pest.scale, 3),
                    "path": path_dict,
                }
            )
        return raw


def save_layout_spec(layout_spec: LayoutSpec, path: Path) -> None:
    """Write a layout spec to disk as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(layout_spec.to_dict(), handle, indent=2)


def load_layout_spec(path: Path) -> dict[str, object]:
    """Load a previously generated layout spec."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_layout_decisions(layout_spec: LayoutSpec) -> dict[str, object]:
    """Summarize the main layout decisions for debugging and iteration."""
    fridge = next((fixture for fixture in layout_spec.fixtures if fixture.kind == "fridge"), None)
    island = next((fixture for fixture in layout_spec.fixtures if fixture.kind == "table"), None)
    return {
        "source_photo": layout_spec.source_photo,
        "photo_size": [int(layout_spec.photo_size[0]), int(layout_spec.photo_size[1])],
        "photo_cues": photo_cues_to_dict(layout_spec.photo_cues),
        "room": {
            "width": round(layout_spec.room.width, 3),
            "depth": round(layout_spec.room.depth, 3),
            "height": round(layout_spec.room.height, 3),
        },
        "camera": {
            "location": _rounded_triplet(layout_spec.camera.location),
            "rotation_deg": _rounded_triplet(layout_spec.camera.rotation_deg),
            "lens_mm": round(layout_spec.camera.lens_mm, 3),
        },
        "fixture_summary": {
            "fixture_count": len(layout_spec.fixtures),
            "fridge_side": "left"
            if fridge and fridge.center[0] < 0
            else "right"
            if fridge
            else "missing",
            "has_island": island is not None,
            "island_size": _rounded_triplet(island.size) if island else None,
        },
        "pest_types": [pest.pest_type for pest in layout_spec.pests],
    }


def save_layout_diagnostics(layout_spec: LayoutSpec, path: Path) -> None:
    """Persist a human-readable diagnostics file for layout inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summarize_layout_decisions(layout_spec), handle, indent=2)


def extract_photo_cues(photo_path: Path) -> PhotoCuesSpec:
    """Extract simple visual cues from the input kitchen photo."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to analyze the kitchen photo.") from exc

    with Image.open(photo_path) as image:
        image = image.convert("RGB").resize((96, 96))
        width, height = image.size
        pixels = list(image.getdata())

    def rgb(row: int, col: int) -> tuple[float, float, float]:
        red, green, blue = pixels[row * width + col]
        return red / 255.0, green / 255.0, blue / 255.0

    def luma(row: int, col: int) -> float:
        red, green, blue = rgb(row, col)
        return 0.299 * red + 0.587 * green + 0.114 * blue

    def band_mean(row_start: int, row_end: int, col_start: int = 0, col_end: int = 96) -> float:
        values = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                values.append(luma(row, col))
        return _mean(values)

    brightness_top = band_mean(0, 24)
    brightness_mid = band_mean(24, 64)
    brightness_bottom = band_mean(64, 96)
    left_brightness = band_mean(0, 96, 0, 32)
    center_brightness = band_mean(0, 96, 32, 64)
    right_brightness = band_mean(0, 96, 64, 96)

    row_edges = []
    for row in range(1, height):
        diffs = [abs(luma(row, col) - luma(row - 1, col)) for col in range(width)]
        row_edges.append(_mean(diffs))
    lower_half_edges = row_edges[height // 3 :]
    floor_index = lower_half_edges.index(max(lower_half_edges)) + height // 3
    floor_line_ratio = floor_index / max(1, height - 1)

    pixel_brightness = [luma(row, col) for row in range(height) for col in range(width)]
    contrast_score = _clamp(max(pixel_brightness) - min(pixel_brightness), 0.0, 1.0)

    edge_strengths = []
    for row in range(height - 1):
        for col in range(width - 1):
            horizontal = abs(luma(row, col + 1) - luma(row, col))
            vertical = abs(luma(row + 1, col) - luma(row, col))
            edge_strengths.append((horizontal + vertical) / 2.0)
    clutter_score = _clamp(_mean(edge_strengths) * 4.0, 0.0, 1.0)

    warm_values = []
    for row in range(height):
        for col in range(width):
            red, green, blue = rgb(row, col)
            warm_values.append((red - blue + (red - green) * 0.5 + 1.0) / 2.0)
    warm_bias = _clamp(_mean(warm_values), 0.0, 1.0)

    return PhotoCuesSpec(
        brightness_top=brightness_top,
        brightness_mid=brightness_mid,
        brightness_bottom=brightness_bottom,
        left_brightness=left_brightness,
        center_brightness=center_brightness,
        right_brightness=right_brightness,
        floor_line_ratio=floor_line_ratio,
        clutter_score=clutter_score,
        warm_bias=warm_bias,
        contrast_score=contrast_score,
    )


def _build_room_spec(photo_size: tuple[int, int], photo_cues: PhotoCuesSpec) -> RoomSpec:
    width_px, height_px = photo_size
    aspect_ratio = width_px / max(height_px, 1)
    openness = 1.0 - photo_cues.clutter_score
    floor_weight = _clamp((photo_cues.floor_line_ratio - 0.45) / 0.4, 0.0, 1.0)

    room_width = 5.5 + (aspect_ratio - 1.1) * 1.2 + openness * 1.0
    room_depth = 4.6 + floor_weight * 2.2 + (1.0 - openness) * 0.6
    room_height = 2.7 + photo_cues.brightness_top * 0.35
    return RoomSpec(
        width=round(_clamp(room_width, 4.8, 8.4), 3),
        depth=round(_clamp(room_depth, 4.2, 7.5), 3),
        height=round(_clamp(room_height, 2.5, 3.4), 3),
    )


def _build_camera_spec(room: RoomSpec, photo_cues: PhotoCuesSpec) -> CameraSpec:
    horizontal_bias = photo_cues.right_brightness - photo_cues.left_brightness
    location_x = horizontal_bias * room.width * 0.15
    location_y = -(room.depth + 0.9 + photo_cues.clutter_score * 0.8)
    location_z = room.height + 1.2 + (1.0 - photo_cues.floor_line_ratio) * 1.3
    tilt = 48.0 + photo_cues.floor_line_ratio * 18.0
    yaw = horizontal_bias * 8.0
    lens = 28.0 + (1.0 - photo_cues.clutter_score) * 6.0 + photo_cues.contrast_score * 2.0
    return CameraSpec(
        location=(round(location_x, 3), round(location_y, 3), round(location_z, 3)),
        rotation_deg=(round(tilt, 3), 0.0, round(yaw, 3)),
        lens_mm=round(_clamp(lens, 26.0, 38.0), 3),
    )


def _build_light_specs(room: RoomSpec, photo_cues: PhotoCuesSpec) -> list[LightSpec]:
    warm_strength = photo_cues.warm_bias
    fill_bias = photo_cues.left_brightness - photo_cues.right_brightness
    key_energy = 340.0 + photo_cues.brightness_top * 220.0
    fill_energy = 120.0 + photo_cues.brightness_mid * 120.0
    return [
        LightSpec(
            name="key",
            kind="AREA",
            location=(0.0, 0.0, room.height + 1.5),
            energy=round(key_energy, 3),
            color=(
                1.0,
                round(0.9 + warm_strength * 0.06, 3),
                round(0.84 + warm_strength * 0.08, 3),
            ),
        ),
        LightSpec(
            name="fill",
            kind="AREA",
            location=(
                round((-0.35 - fill_bias * 0.15) * room.width, 3),
                round(-0.35 * room.depth, 3),
                round(room.height + 0.55, 3),
            ),
            energy=round(fill_energy, 3),
            color=(round(0.85 + (1.0 - warm_strength) * 0.08, 3), 0.92, 1.0),
        ),
        LightSpec(
            name="rim",
            kind="POINT",
            location=(
                round(room.width * 0.28, 3),
                round(room.depth * 0.42, 3),
                round(room.height + 0.15, 3),
            ),
            energy=round(50.0 + photo_cues.contrast_score * 60.0, 3),
            color=(1.0, 0.93, round(0.82 + warm_strength * 0.08, 3)),
        ),
    ]


def _build_fixture_specs(room: RoomSpec, photo_cues: PhotoCuesSpec) -> list[FixtureSpec]:
    wall_y = room.depth / 2.0
    counter_depth = 0.62 + photo_cues.clutter_score * 0.1
    cabinet_height = 0.9
    left_weight = _clamp(photo_cues.left_brightness * 1.1, 0.25, 0.9)
    right_weight = _clamp(photo_cues.right_brightness * 1.1, 0.25, 0.9)
    center_weight = _clamp(photo_cues.center_brightness * 1.1, 0.25, 0.9)
    warmth = photo_cues.warm_bias

    wood_base = (0.28 + warmth * 0.18, 0.2 + warmth * 0.12, 0.14 + warmth * 0.08)
    counter_color = (0.48 + photo_cues.contrast_score * 0.1, 0.47, 0.45)

    fixtures = [
        FixtureSpec(
            name="counter_back",
            kind="counter",
            center=(0.0, wall_y - counter_depth / 2.0, cabinet_height),
            size=(room.width * (0.64 + center_weight * 0.16), counter_depth, 0.08),
            color=counter_color,
        ),
        FixtureSpec(
            name="cabinet_left",
            kind="cabinet",
            center=(
                -room.width * (0.23 + left_weight * 0.04),
                wall_y - counter_depth / 2.0,
                cabinet_height / 2.0,
            ),
            size=(1.0 + left_weight * 0.9, counter_depth, cabinet_height),
            color=wood_base,
        ),
        FixtureSpec(
            name="cabinet_right",
            kind="cabinet",
            center=(
                room.width * (0.22 + right_weight * 0.04),
                wall_y - counter_depth / 2.0,
                cabinet_height / 2.0,
            ),
            size=(1.1 + right_weight * 1.0, counter_depth, cabinet_height),
            color=(wood_base[0] * 0.92, wood_base[1] * 0.92, wood_base[2] * 0.92),
        ),
        FixtureSpec(
            name="fridge",
            kind="fridge",
            center=(
                -room.width
                * (
                    0.42
                    if photo_cues.left_brightness < photo_cues.right_brightness
                    else -0.42
                ),
                wall_y - 0.42,
                1.05,
            ),
            size=(0.9, 0.85, 2.1),
            color=(0.86, 0.86, 0.88),
        ),
        FixtureSpec(
            name="island_table",
            kind="table",
            center=(
                room.width * 0.12,
                -room.depth * (0.08 + photo_cues.clutter_score * 0.08),
                0.42,
            ),
            size=(0.9 + center_weight * 0.45, 0.6 + photo_cues.clutter_score * 0.18, 0.84),
            color=(wood_base[0] * 1.08, wood_base[1] * 1.02, wood_base[2]),
        ),
    ]
    return fixtures


def _lateral_point(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t: float,
    lateral: float,
) -> tuple[float, float, float]:
    """Interpolate along start→end and offset perpendicular in the XY plane."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return (start[0], start[1], start[2])
    px, py = -dy / length, dx / length
    x = start[0] + t * dx + lateral * px
    y = start[1] + t * dy + lateral * py
    z = start[2] + t * (end[2] - start[2])
    return (round(x, 3), round(y, 3), round(z, 3))


def _build_pest_specs(
    room: RoomSpec,
    photo_cues: PhotoCuesSpec,
    pest_types: Sequence[str],
    scene_seed: int,
) -> list[PestInstanceSpec]:
    rng = random.Random(scene_seed)
    # Keep pests in the near/lower part of the camera frustum. This makes photo
    # background renders look grounded instead of placing pests on counters.
    anchor_y = -room.depth * (0.28 + photo_cues.floor_line_ratio * 0.12)
    base_paths = [
        (
            (-room.width * 0.34, anchor_y, 0.0),
            (room.width * 0.12, anchor_y + room.depth * 0.07, 0.0),
        ),
        (
            (room.width * 0.26, anchor_y + room.depth * 0.05, 0.0),
            (-room.width * 0.08, anchor_y + room.depth * 0.12, 0.0),
        ),
        (
            (-room.width * 0.18, anchor_y + room.depth * 0.15, 0.0),
            (room.width * 0.24, anchor_y + room.depth * 0.18, 0.0),
        ),
    ]

    pests: list[PestInstanceSpec] = []
    for index, pest_type in enumerate(pest_types):
        start, end = base_paths[index % len(base_paths)]
        scale = 0.82 if pest_type == "mouse" else 1.05 if pest_type == "rat" else 0.45
        elevation = 0.08 if pest_type in {"mouse", "rat"} else 0.03
        jitter_x = rng.uniform(-0.14, 0.14)
        jitter_y = rng.uniform(-0.12, 0.12)
        s = (
            round(start[0] + jitter_x, 3),
            round(start[1] + jitter_y, 3),
            elevation,
        )
        e = (round(end[0] + jitter_x, 3), round(end[1] + jitter_y, 3), elevation)
        w_mag = room.width * (0.1 + 0.04 * (index % 3)) * (0.55 + photo_cues.clutter_score * 0.25)
        sign = 1.0 if index % 2 == 0 else -1.0
        w1 = _lateral_point(s, e, 1.0 / 3.0, sign * w_mag)
        w2 = _lateral_point(s, e, 2.0 / 3.0, -sign * w_mag)
        pests.append(
            PestInstanceSpec(
                pest_id=f"{pest_type}_{index + 1}",
                pest_type=pest_type,
                scale=scale,
                path=PestPathSpec(
                    start=s,
                    end=e,
                    elevation=elevation,
                    waypoints=(w1, w2),
                ),
            )
        )
    return pests


def build_layout_spec(
    photo_path: Path,
    photo_size: tuple[int, int],
    photo_cues: PhotoCuesSpec,
    pest_types: Sequence[str],
    scene_seed: int,
) -> LayoutSpec:
    """Infer a kitchen layout from extracted photo cues."""
    room = _build_room_spec(photo_size, photo_cues)
    camera = _build_camera_spec(room, photo_cues)
    lights = _build_light_specs(room, photo_cues)
    fixtures = _build_fixture_specs(room, photo_cues)
    pests = _build_pest_specs(room, photo_cues, pest_types, scene_seed)
    return LayoutSpec(
        schema_version="0.2.1",
        source_photo=str(photo_path),
        photo_size=photo_size,
        photo_cues=photo_cues,
        room=room,
        camera=camera,
        lights=lights,
        fixtures=fixtures,
        pests=pests,
    )
