"""Blender entrypoint that turns a layout spec into rendered frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
except ModuleNotFoundError as exc:  # pragma: no cover - runs only inside Blender
    raise SystemExit("This script must be executed inside Blender.") from exc


def parse_args() -> argparse.Namespace:
    """Parse arguments passed after Blender's `--` separator."""
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-spec", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    return parser.parse_args(argv)


def load_layout(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras]:
        for block in list(collection):
            collection.remove(block)


def make_material(name: str, color: list[float], roughness: float = 0.5):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def build_room(layout: dict) -> None:
    room = layout["room"]
    width = room["width"]
    depth = room["depth"]
    height = room["height"]

    floor_material = make_material("Floor", [0.86, 0.84, 0.8], roughness=0.55)
    wall_material = make_material("Wall", [0.93, 0.91, 0.88], roughness=0.7)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.scale = (width / 2.0, depth / 2.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    floor.data.materials.append(floor_material)

    wall_specs = [
        (
            (0, depth / 2.0, height / 2.0),
            (width / 2.0, height / 2.0, 1.0),
            (math.radians(90), 0, 0),
        ),
        (
            (-width / 2.0, 0, height / 2.0),
            (depth / 2.0, height / 2.0, 1.0),
            (0, math.radians(-90), 0),
        ),
        (
            (width / 2.0, 0, height / 2.0),
            (depth / 2.0, height / 2.0, 1.0),
            (0, math.radians(90), 0),
        ),
    ]
    for location, scale, rotation in wall_specs:
        bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=rotation)
        wall = bpy.context.active_object
        wall.scale = scale
        bpy.ops.object.transform_apply(scale=True)
        wall.data.materials.append(wall_material)


def build_fixture(fixture: dict) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1, location=tuple(fixture["center"]))
    obj = bpy.context.active_object
    obj.name = fixture["name"]
    obj.scale = tuple(value / 2.0 for value in fixture["size"])
    bpy.ops.object.transform_apply(scale=True)
    obj.data.materials.append(make_material(f"{fixture['name']}_mat", fixture["color"], 0.45))


def build_pest(pest: dict):
    pest_type = pest["pest_type"]
    location = tuple(pest["path"]["start"])

    if pest_type == "cockroach":
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.08,
            location=location,
            segments=16,
            ring_count=8,
        )
        obj = bpy.context.active_object
        obj.scale = (1.3 * pest["scale"], 0.8 * pest["scale"], 0.28 * pest["scale"])
        color = [0.34, 0.16, 0.08]
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.1,
            location=location,
            segments=20,
            ring_count=10,
        )
        obj = bpy.context.active_object
        if pest_type == "rat":
            obj.scale = (2.2 * pest["scale"], 1.2 * pest["scale"], 0.9 * pest["scale"])
            color = [0.28, 0.27, 0.26]
        else:
            obj.scale = (1.8 * pest["scale"], 1.0 * pest["scale"], 0.75 * pest["scale"])
            color = [0.48, 0.43, 0.38]

    obj.name = pest["pest_id"]
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(make_material(f"{pest['pest_id']}_mat", color, 0.7))
    return obj


def animate_pest(obj, pest: dict, frame_end: int) -> None:
    obj.location = tuple(pest["path"]["start"])
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = tuple(pest["path"]["end"])
    obj.keyframe_insert(data_path="location", frame=frame_end)


def setup_camera(layout: dict):
    camera_spec = layout["camera"]
    rotation = tuple(math.radians(value) for value in camera_spec["rotation_deg"])
    bpy.ops.object.camera_add(location=tuple(camera_spec["location"]), rotation=rotation)
    camera = bpy.context.active_object
    camera.data.lens = camera_spec["lens_mm"]
    bpy.context.scene.camera = camera
    return camera


def setup_lighting(layout: dict) -> None:
    for light in layout["lights"]:
        bpy.ops.object.light_add(type=light["kind"], location=tuple(light["location"]))
        obj = bpy.context.active_object
        obj.name = light["name"]
        obj.data.energy = light["energy"]
        obj.data.color = tuple(light["color"])
        if light["kind"] == "AREA":
            obj.data.size = 3.0


def setup_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = args.fps
    scene.frame_start = 1
    scene.frame_end = args.fps * args.seconds
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Contrast"


def compute_bbox(obj, camera, scene, width: int, height: int):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    corners_world = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]

    xs = []
    ys = []
    for corner in corners_world:
        ndc = world_to_camera_view(scene, camera, corner)
        if ndc.z < 0:
            return None
        xs.append(ndc.x * width)
        ys.append((1.0 - ndc.y) * height)

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(width), max(xs))
    y_max = min(float(height), max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "x_min": round(x_min, 1),
        "y_min": round(y_min, 1),
        "x_max": round(x_max, 1),
        "y_max": round(y_max, 1),
        "width": round(x_max - x_min, 1),
        "height": round(y_max - y_min, 1),
    }


def main() -> None:
    args = parse_args()
    layout = load_layout(Path(args.layout_spec))
    frames_dir = Path(args.frames_dir)
    annotations_path = Path(args.annotations)
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    setup_render(args)
    build_room(layout)
    for fixture in layout["fixtures"]:
        build_fixture(fixture)
    camera = setup_camera(layout)
    setup_lighting(layout)

    pest_objects = []
    frame_end = args.fps * args.seconds
    for pest in layout["pests"]:
        obj = build_pest(pest)
        animate_pest(obj, pest, frame_end)
        pest_objects.append((pest["pest_type"], obj))

    scene = bpy.context.scene
    annotations = []
    for frame in range(1, frame_end + 1):
        scene.frame_set(frame)
        frame_path = frames_dir / f"frame_{frame:05d}.png"
        scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)

        frame_pests = []
        for pest_type, obj in pest_objects:
            bbox = compute_bbox(obj, camera, scene, args.width, args.height)
            if bbox is None:
                continue
            frame_pests.append({"label": pest_type, "bbox": bbox})

        annotations.append(
            {
                "frame": frame,
                "file": str(frame_path.resolve()),
                "pests": frame_pests,
            }
        )

    with annotations_path.open("w", encoding="utf-8") as handle:
        json.dump(annotations, handle, indent=2)

    print(f"Rendered {frame_end} frames -> {frames_dir}")
    print(f"Annotations -> {annotations_path}")


if __name__ == "__main__":
    main()
