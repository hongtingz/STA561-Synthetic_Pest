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
    parser.add_argument("--render-device", default="CPU")
    parser.add_argument("--compute-backend", default="AUTO")
    parser.add_argument("--photo-background", default="false")
    parser.add_argument("--pest-asset-style", default="procedural_v2")
    return parser.parse_args(argv)


def load_layout(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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


def make_image_material(name: str, image_path: Path):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(image_path))
    links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    if "Emission Color" in bsdf.inputs:
        links.new(image_node.outputs["Color"], bsdf.inputs["Emission Color"])
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.35
    return material


def make_pest_material(name: str, color: list[float], roughness: float = 0.78):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    return material


def add_photo_background(layout: dict, camera, scene) -> None:
    """Place the source kitchen photo on a renderable plane behind the pests."""
    photo_path = Path(layout.get("source_photo", ""))
    if not photo_path.exists():
        print(f"Photo background missing, keeping synthetic room only: {photo_path}")
        return

    distance = max(12.0, float(layout["room"]["depth"]) * 2.4)
    frame = camera.data.view_frame(scene=scene)
    xs = [corner.x for corner in frame]
    ys = [corner.y for corner in frame]
    width = (max(xs) - min(xs)) * distance
    height = (max(ys) - min(ys)) * distance

    local_center = Vector((0.0, 0.0, -distance))
    world_center = camera.matrix_world @ local_center
    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=world_center,
        rotation=camera.rotation_euler,
    )
    plane = bpy.context.active_object
    plane.name = "source_photo_background"
    plane.scale = (width / 2.0, height / 2.0, 1.0)
    plane.data.materials.append(make_image_material("source_photo_mat", photo_path))
    plane.hide_select = True
    print(f"Using source photo background: {photo_path}")


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


def _path_yaw(pest: dict) -> float:
    start = pest["path"]["start"]
    end = pest["path"]["end"]
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _create_root(name: str, pest: dict):
    root = bpy.data.objects.new(name=name, object_data=None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.18
    root.location = tuple(pest["path"]["start"])
    root.rotation_euler = (0.0, 0.0, _path_yaw(pest))
    bpy.context.collection.objects.link(root)
    return root


def _add_child_ellipsoid(
    root,
    name: str,
    material,
    *,
    radius: float,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    segments: int = 24,
    ring_count: int = 12,
):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(0.0, 0.0, 0.0),
        segments=segments,
        ring_count=ring_count,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.parent = root
    obj.location = location
    obj.scale = scale
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(material)
    return obj


def _add_child_cylinder(
    root,
    name: str,
    material,
    *,
    radius: float,
    depth: float,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=12,
        radius=radius,
        depth=depth,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.parent = root
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(material)
    return obj


def build_rodent_pest(pest: dict, *, is_rat: bool):
    root = _create_root(pest["pest_id"], pest)
    scale = float(pest["scale"])
    if is_rat:
        body_color = [0.25, 0.24, 0.23]
        belly_color = [0.43, 0.39, 0.35]
        body_len = 0.24 * scale
        body_w = 0.105 * scale
        body_h = 0.075 * scale
        head_size = 0.065 * scale
        tail_len = 0.24 * scale
    else:
        body_color = [0.45, 0.41, 0.36]
        belly_color = [0.62, 0.56, 0.49]
        body_len = 0.16 * scale
        body_w = 0.075 * scale
        body_h = 0.055 * scale
        head_size = 0.047 * scale
        tail_len = 0.16 * scale

    body_mat = make_pest_material(f"{pest['pest_id']}_fur_mat", body_color)
    belly_mat = make_pest_material(f"{pest['pest_id']}_belly_mat", belly_color)
    dark_mat = make_pest_material(f"{pest['pest_id']}_dark_mat", [0.05, 0.045, 0.04])
    pink_mat = make_pest_material(f"{pest['pest_id']}_tail_mat", [0.58, 0.36, 0.33])

    _add_child_ellipsoid(
        root,
        f"{pest['pest_id']}_body",
        body_mat,
        radius=1.0,
        location=(0.0, 0.0, 0.02 * scale),
        scale=(body_len, body_w, body_h),
    )
    _add_child_ellipsoid(
        root,
        f"{pest['pest_id']}_belly",
        belly_mat,
        radius=1.0,
        location=(0.02 * scale, 0.0, -0.006 * scale),
        scale=(body_len * 0.72, body_w * 0.72, body_h * 0.33),
        segments=20,
        ring_count=8,
    )
    _add_child_ellipsoid(
        root,
        f"{pest['pest_id']}_head",
        body_mat,
        radius=1.0,
        location=(body_len * 0.9, 0.0, 0.036 * scale),
        scale=(head_size, head_size * 0.8, head_size * 0.72),
    )
    for side in [-1.0, 1.0]:
        _add_child_ellipsoid(
            root,
            f"{pest['pest_id']}_ear_{side:+.0f}",
            body_mat,
            radius=1.0,
            location=(body_len * 0.91, side * head_size * 0.55, head_size * 1.0),
            scale=(head_size * 0.22, head_size * 0.18, head_size * 0.28),
            segments=12,
            ring_count=6,
        )
        _add_child_ellipsoid(
            root,
            f"{pest['pest_id']}_eye_{side:+.0f}",
            dark_mat,
            radius=1.0,
            location=(body_len * 1.25, side * head_size * 0.35, head_size * 0.65),
            scale=(head_size * 0.08, head_size * 0.06, head_size * 0.06),
            segments=8,
            ring_count=4,
        )
        for leg_x in [-body_len * 0.35, body_len * 0.45]:
            _add_child_cylinder(
                root,
                f"{pest['pest_id']}_leg_{leg_x:.2f}_{side:+.0f}",
                dark_mat,
                radius=0.009 * scale,
                depth=0.11 * scale,
                location=(leg_x, side * body_w * 0.62, -body_h * 0.54),
                rotation=(math.radians(82), 0.0, math.radians(8 * side)),
            )

    _add_child_cylinder(
        root,
        f"{pest['pest_id']}_tail",
        pink_mat,
        radius=0.012 * scale,
        depth=tail_len,
        location=(-body_len * 0.95, 0.0, 0.006 * scale),
        rotation=(0.0, math.radians(86), 0.0),
    )
    return root


def build_cockroach_pest(pest: dict):
    root = _create_root(pest["pest_id"], pest)
    scale = float(pest["scale"])
    shell_mat = make_pest_material(f"{pest['pest_id']}_shell_mat", [0.20, 0.08, 0.035], 0.62)
    stripe_mat = make_pest_material(f"{pest['pest_id']}_stripe_mat", [0.42, 0.18, 0.07], 0.55)
    leg_mat = make_pest_material(f"{pest['pest_id']}_leg_mat", [0.08, 0.035, 0.018], 0.72)

    _add_child_ellipsoid(
        root,
        f"{pest['pest_id']}_abdomen",
        shell_mat,
        radius=1.0,
        location=(-0.025 * scale, 0.0, 0.018 * scale),
        scale=(0.105 * scale, 0.052 * scale, 0.024 * scale),
        segments=28,
        ring_count=10,
    )
    _add_child_ellipsoid(
        root,
        f"{pest['pest_id']}_thorax",
        stripe_mat,
        radius=1.0,
        location=(0.07 * scale, 0.0, 0.021 * scale),
        scale=(0.058 * scale, 0.046 * scale, 0.021 * scale),
        segments=20,
        ring_count=8,
    )
    _add_child_ellipsoid(
        root,
        f"{pest['pest_id']}_head",
        shell_mat,
        radius=1.0,
        location=(0.125 * scale, 0.0, 0.02 * scale),
        scale=(0.026 * scale, 0.03 * scale, 0.017 * scale),
        segments=16,
        ring_count=8,
    )

    for side in [-1.0, 1.0]:
        for index, leg_x in enumerate([-0.045, 0.02, 0.085]):
            angle = math.radians(52 + index * 9)
            _add_child_cylinder(
                root,
                f"{pest['pest_id']}_leg_{index}_{side:+.0f}",
                leg_mat,
                radius=0.0045 * scale,
                depth=0.12 * scale,
                location=(leg_x * scale, side * 0.067 * scale, 0.001 * scale),
                rotation=(math.radians(90), angle * side, math.radians(18 * side)),
            )
        _add_child_cylinder(
            root,
            f"{pest['pest_id']}_antenna_{side:+.0f}",
            leg_mat,
            radius=0.0028 * scale,
            depth=0.12 * scale,
            location=(0.158 * scale, side * 0.035 * scale, 0.031 * scale),
            rotation=(math.radians(82), math.radians(74) * side, math.radians(25 * side)),
        )
    return root


def build_simple_pest(pest: dict):
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


def build_pest(pest: dict, asset_style: str):
    if asset_style == "simple":
        return build_simple_pest(pest)
    if pest["pest_type"] == "cockroach":
        return build_cockroach_pest(pest)
    return build_rodent_pest(pest, is_rat=pest["pest_type"] == "rat")


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


def _configure_cycles_gpu(compute_backend: str) -> bool:
    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        print("Cycles add-on preferences unavailable; falling back to CPU.")
        return False

    preferences = cycles_addon.preferences
    backend = compute_backend.upper()
    if backend != "AUTO":
        try:
            preferences.compute_device_type = backend
        except (TypeError, ValueError) as exc:
            print(f"Unsupported compute backend '{backend}'; falling back to CPU: {exc}")
            return False

    try:
        preferences.get_devices()
    except Exception as exc:  # pragma: no cover - Blender runtime dependent
        print(f"Could not enumerate Cycles devices; falling back to CPU: {exc}")
        return False

    gpu_count = 0
    for device in getattr(preferences, "devices", []):
        if getattr(device, "type", "CPU") == "CPU":
            device.use = False
            continue
        device.use = True
        gpu_count += 1

    if gpu_count == 0:
        print("No GPU devices detected for Cycles; falling back to CPU.")
        return False

    active_backend = getattr(preferences, "compute_device_type", backend)
    print(f"Configured Cycles GPU rendering: backend={active_backend} devices={gpu_count}")
    return True


def setup_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
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

    render_device = args.render_device.upper()
    if render_device == "GPU" and _configure_cycles_gpu(args.compute_backend):
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"
        print("Configured Cycles rendering on CPU.")


def _mesh_objects_for_bbox(obj) -> list:
    objects = []
    candidates = [obj, *getattr(obj, "children_recursive", [])]
    for candidate in candidates:
        if getattr(candidate, "type", None) == "MESH":
            objects.append(candidate)
    return objects


def compute_bbox(obj, camera, scene, width: int, height: int):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners_world = []
    for mesh_obj in _mesh_objects_for_bbox(obj):
        obj_eval = mesh_obj.evaluated_get(depsgraph)
        corners_world.extend(
            obj_eval.matrix_world @ Vector(corner)
            for corner in obj_eval.bound_box
        )
    if not corners_world:
        return None

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
    camera = setup_camera(layout)
    if parse_bool(args.photo_background):
        add_photo_background(layout, camera, bpy.context.scene)
    else:
        build_room(layout)
        for fixture in layout["fixtures"]:
            build_fixture(fixture)
    setup_lighting(layout)

    pest_objects = []
    frame_end = args.fps * args.seconds
    for pest in layout["pests"]:
        obj = build_pest(pest, args.pest_asset_style)
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
