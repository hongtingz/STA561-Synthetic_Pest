"""
Pest Detection Demo — Blender Script
=====================================
Creates a simple kitchen scene with an animated mouse and renders
60 frames with per-frame bounding box annotations.

Run from Blender Scripting tab (open this file and hit Run Script),
or headless:
    blender --background --python demo/render_demo.py
f
Output:
    output/frames/frame_XXXX.png   — rendered frames
    output/annotations.json        — per-frame bbox annotations
"""

import json
import math
import os

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

# ── Config ────────────────────────────────────────────────────────────────────
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running from Blender GUI scripting tab (pasted/unsaved) — fall back to CWD
    SCRIPT_DIR = os.path.join(os.getcwd(), "demo")
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output"))
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
NUM_FRAMES = 60  # 2 seconds at 30 fps
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

os.makedirs(FRAMES_DIR, exist_ok=True)


# ── Scene Setup ───────────────────────────────────────────────────────────────
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Remove orphaned data (snapshot the list to avoid mutating while iterating)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)


def make_material(name, color_rgba):
    mat = bpy.data.materials.new(name=name)
    # Blender 5.0+: nodes enabled by default, no need for mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color_rgba
    return mat


def build_kitchen():
    # Floor — large plane at z=0
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "KitchenFloor"
    floor.data.materials.append(make_material("FloorMat", (0.78, 0.76, 0.72, 1.0)))

    # Back wall — vertical plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 5, 2.5))
    wall = bpy.context.active_object
    wall.name = "BackWall"
    wall.rotation_euler = (math.radians(90), 0, 0)
    wall.data.materials.append(make_material("WallMat", (0.94, 0.92, 0.88, 1.0)))

    # Simple baseboard (thin box along back wall base)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 4.95, 0.1))
    board = bpy.context.active_object
    board.name = "Baseboard"
    board.scale = (5, 0.05, 0.1)
    bpy.ops.object.transform_apply(scale=True)
    board.data.materials.append(make_material("BoardMat", (0.85, 0.83, 0.80, 1.0)))


def build_mouse():
    """Placeholder mouse: elongated UV sphere, dark brown."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.15,
        segments=16,
        ring_count=8,
        location=(-1.5, 0, 0.15),
    )
    mouse = bpy.context.active_object
    mouse.name = "Mouse"
    mouse.scale = (1.6, 1.0, 0.7)
    bpy.ops.object.transform_apply(scale=True)
    mouse.data.materials.append(make_material("MouseMat", (0.18, 0.14, 0.12, 1.0)))
    return mouse


def animate_mouse(mouse):
    """Walk from left to right across the kitchen floor."""
    scene = bpy.context.scene

    mouse.location = (-1.5, 0, 0.15)
    mouse.keyframe_insert(data_path="location", frame=1)

    # Mid-point slight curve for realism
    mouse.location = (0, -0.3, 0.15)
    mouse.keyframe_insert(data_path="location", frame=NUM_FRAMES // 2)

    mouse.location = (1.5, 0, 0.15)
    mouse.keyframe_insert(data_path="location", frame=NUM_FRAMES)

    # BEZIER is the default interpolation in Blender 5.0


def setup_camera():
    bpy.ops.object.camera_add(location=(0, -6, 5))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(55), 0, 0)
    bpy.context.scene.camera = cam
    return cam


def setup_lighting():
    # Main overhead area light
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 6))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 600
    key.data.size = 5

    # Fill point light
    bpy.ops.object.light_add(type="POINT", location=(-4, -3, 4))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 120


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.image_settings.file_format = "PNG"
    scene.frame_start = 1
    scene.frame_end = NUM_FRAMES
    scene.render.fps = 30


# ── Bounding Box ──────────────────────────────────────────────────────────────
def get_2d_bbox(obj, camera, scene):
    """
    Project the object's 8 world-space bounding-box corners through the
    camera and return a 2D axis-aligned bounding box in pixel coordinates.
    Returns None if the object is entirely behind the camera.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    corners_world = [obj_eval.matrix_world @ Vector(c) for c in obj_eval.bound_box]

    xs, ys = [], []
    for co in corners_world:
        ndc = world_to_camera_view(scene, camera, co)
        if ndc.z < 0:
            return None  # behind camera
        xs.append(ndc.x * RENDER_WIDTH)
        ys.append((1.0 - ndc.y) * RENDER_HEIGHT)  # flip Y axis

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(RENDER_WIDTH), max(xs))
    y_max = min(float(RENDER_HEIGHT), max(ys))

    return {
        "x_min": round(x_min, 1),
        "y_min": round(y_min, 1),
        "x_max": round(x_max, 1),
        "y_max": round(y_max, 1),
        "width": round(x_max - x_min, 1),
        "height": round(y_max - y_min, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Pest Detection Demo ===\n")

    clear_scene()
    build_kitchen()
    mouse = build_mouse()
    animate_mouse(mouse)
    camera = setup_camera()
    setup_lighting()
    setup_render()

    scene = bpy.context.scene
    annotations = []

    for frame in range(1, NUM_FRAMES + 1):
        scene.frame_set(frame)

        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame:04d}.png")
        scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)

        bbox = get_2d_bbox(mouse, camera, scene)
        pest_entry = {"label": "mouse", "bbox": bbox} if bbox else None

        annotations.append(
            {
                "frame": frame,
                "file": os.path.abspath(frame_path),
                "pests": [pest_entry] if pest_entry else [],
            }
        )

        print(f"  Frame {frame:03d}/{NUM_FRAMES}  bbox={bbox}")

    anno_path = os.path.join(OUTPUT_DIR, "annotations.json")
    with open(anno_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDone! Frames → {FRAMES_DIR}")
    print(f"Annotations → {anno_path}")


main()
