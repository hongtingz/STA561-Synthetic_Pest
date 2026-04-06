"""
Pest Detection Demo — Blender Script
=====================================
Creates a realistic kitchen scene with an animated mouse and renders
60 frames with per-frame bounding box annotations.

Run headless:
    blender --background --python demo/render_demo.py

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
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in list(bpy.data.lights):
        bpy.data.lights.remove(block)
    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block)


# ── Materials ─────────────────────────────────────────────────────────────────
def _get_bsdf(mat):
    """Return the Principled BSDF node from a material."""
    return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def make_floor_material():
    """Procedural kitchen tile floor — checker pattern with roughness variation."""
    mat = bpy.data.materials.new(name="FloorTileMat")
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    bsdf = _get_bsdf(mat)

    # Texture coordinate + mapping
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    # Checker texture for tile pattern
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.82, 0.78, 0.72, 1.0)  # warm beige
    checker.inputs["Color2"].default_value = (0.65, 0.60, 0.55, 1.0)  # darker tile
    checker.inputs["Scale"].default_value = 6.0
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])

    # Slight roughness variation from noise
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 15.0
    noise.inputs["Detail"].default_value = 4.0
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = 0.15
    map_range.inputs["To Max"].default_value = 0.45
    links.new(noise.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], bsdf.inputs["Roughness"])

    # Subtle bump from noise
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.1
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Specular IOR Level"].default_value = 0.3

    return mat


def make_wall_material():
    """Painted drywall — subtle orange-peel bump texture."""
    mat = bpy.data.materials.new(name="WallPaintMat")
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    bsdf = _get_bsdf(mat)

    bsdf.inputs["Base Color"].default_value = (0.92, 0.90, 0.85, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Specular IOR Level"].default_value = 0.1

    # Orange-peel bump
    tex_coord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 80.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.6
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.03
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def make_cabinet_material():
    """Wood-look cabinets with grain texture."""
    mat = bpy.data.materials.new(name="CabinetMat")
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    bsdf = _get_bsdf(mat)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 8.0, 1.0)  # stretch for grain
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    # Wave texture for wood grain
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.inputs["Scale"].default_value = 3.0
    wave.inputs["Distortion"].default_value = 4.0
    wave.inputs["Detail"].default_value = 3.0
    wave.inputs["Detail Scale"].default_value = 1.5
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

    # Color ramp for wood tones
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = (0.25, 0.15, 0.08, 1.0)  # dark wood
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.elements[1].color = (0.40, 0.25, 0.13, 1.0)  # light wood
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Specular IOR Level"].default_value = 0.3

    # Subtle bump from grain
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.05
    links.new(wave.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def make_countertop_material():
    """Granite-like countertop."""
    mat = bpy.data.materials.new(name="CountertopMat")
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    bsdf = _get_bsdf(mat)

    tex_coord = nodes.new("ShaderNodeTexCoord")

    # Voronoi for speckled granite look
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 30.0
    links.new(tex_coord.outputs["Object"], voronoi.inputs["Vector"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 50.0
    noise.inputs["Detail"].default_value = 8.0
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs["Fac"].default_value = 0.5
    mix.inputs["Color1"].default_value = (0.25, 0.25, 0.27, 1.0)  # dark granite
    mix.inputs["Color2"].default_value = (0.45, 0.43, 0.42, 1.0)  # light speckle
    links.new(voronoi.outputs["Distance"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.15
    bsdf.inputs["Specular IOR Level"].default_value = 0.5

    return mat


def make_simple_material(name, color_rgba, roughness=0.5):
    """Basic PBR material with specified roughness."""
    mat = bpy.data.materials.new(name=name)
    bsdf = _get_bsdf(mat)
    bsdf.inputs["Base Color"].default_value = color_rgba
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_mouse_material():
    """Realistic fur-like mouse material with color variation."""
    mat = bpy.data.materials.new(name="MouseMat")
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    bsdf = _get_bsdf(mat)

    tex_coord = nodes.new("ShaderNodeTexCoord")

    # Noise for fur color variation
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 25.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.7
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.12, 0.09, 0.07, 1.0)  # dark brown
    ramp.color_ramp.elements[1].position = 0.65
    ramp.color_ramp.elements[1].color = (0.22, 0.17, 0.13, 1.0)  # lighter brown
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.85  # matte fur
    bsdf.inputs["Specular IOR Level"].default_value = 0.1

    # Fine bump for fur-like texture
    noise2 = nodes.new("ShaderNodeTexNoise")
    noise2.inputs["Scale"].default_value = 80.0
    noise2.inputs["Detail"].default_value = 10.0
    links.new(tex_coord.outputs["Object"], noise2.inputs["Vector"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    links.new(noise2.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


# ── Kitchen Geometry ──────────────────────────────────────────────────────────
def build_kitchen():
    """Build a more detailed kitchen environment."""
    # Floor
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "KitchenFloor"
    floor.data.materials.append(make_floor_material())

    # Back wall
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 5, 2.5))
    wall = bpy.context.active_object
    wall.name = "BackWall"
    wall.rotation_euler = (math.radians(90), 0, 0)
    wall.data.materials.append(make_wall_material())

    # Left wall
    bpy.ops.mesh.primitive_plane_add(size=10, location=(-5, 0, 2.5))
    lwall = bpy.context.active_object
    lwall.name = "LeftWall"
    lwall.rotation_euler = (0, math.radians(-90), 0)
    lwall.data.materials.append(make_wall_material())

    # Baseboard — back wall
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 4.95, 0.06))
    board = bpy.context.active_object
    board.name = "BaseboardBack"
    board.scale = (5, 0.03, 0.06)
    bpy.ops.object.transform_apply(scale=True)
    board.data.materials.append(make_simple_material("BoardMat", (0.9, 0.9, 0.88, 1.0), 0.4))

    # Baseboard — left wall
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-4.95, 0, 0.06))
    board2 = bpy.context.active_object
    board2.name = "BaseboardLeft"
    board2.scale = (0.03, 5, 0.06)
    bpy.ops.object.transform_apply(scale=True)
    board2.data.materials.append(bpy.data.materials["BoardMat"])

    # ── Lower cabinets along back wall ──
    cab_mat = make_cabinet_material()
    counter_mat = make_countertop_material()

    # Left cabinet
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-3.0, 4.5, 0.45))
    cab1 = bpy.context.active_object
    cab1.name = "CabinetLeft"
    cab1.scale = (0.8, 0.4, 0.45)
    bpy.ops.object.transform_apply(scale=True)
    cab1.data.materials.append(cab_mat)

    # Right cabinet
    bpy.ops.mesh.primitive_cube_add(size=1, location=(2.5, 4.5, 0.45))
    cab2 = bpy.context.active_object
    cab2.name = "CabinetRight"
    cab2.scale = (1.2, 0.4, 0.45)
    bpy.ops.object.transform_apply(scale=True)
    cab2.data.materials.append(cab_mat)

    # Countertop (spans across cabinets)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 4.5, 0.92))
    counter = bpy.context.active_object
    counter.name = "Countertop"
    counter.scale = (4.5, 0.45, 0.03)
    bpy.ops.object.transform_apply(scale=True)
    counter.data.materials.append(counter_mat)

    # ── Upper cabinets ──
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-2.5, 4.7, 2.0))
    ucab1 = bpy.context.active_object
    ucab1.name = "UpperCabinetLeft"
    ucab1.scale = (0.7, 0.25, 0.45)
    bpy.ops.object.transform_apply(scale=True)
    ucab1.data.materials.append(cab_mat)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(2.0, 4.7, 2.0))
    ucab2 = bpy.context.active_object
    ucab2.name = "UpperCabinetRight"
    ucab2.scale = (1.0, 0.25, 0.45)
    bpy.ops.object.transform_apply(scale=True)
    ucab2.data.materials.append(cab_mat)

    # ── Fridge (tall box on the left) ──
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-4.2, 4.0, 1.0))
    fridge = bpy.context.active_object
    fridge.name = "Fridge"
    fridge.scale = (0.5, 0.5, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    fridge.data.materials.append(
        make_simple_material("FridgeMat", (0.85, 0.85, 0.87, 1.0), 0.2)
    )

    # ── Small table/shelf near camera side ──
    bpy.ops.mesh.primitive_cube_add(size=1, location=(2.0, 1.0, 0.35))
    table = bpy.context.active_object
    table.name = "SmallTable"
    table.scale = (0.5, 0.4, 0.35)
    bpy.ops.object.transform_apply(scale=True)
    table.data.materials.append(cab_mat)

    # Table legs
    leg_mat = make_simple_material("LegMat", (0.3, 0.3, 0.3, 1.0), 0.6)
    for dx, dy in [(-0.4, -0.3), (0.4, -0.3), (-0.4, 0.3), (0.4, 0.3)]:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.025, depth=0.7,
            location=(2.0 + dx, 1.0 + dy, 0.35)
        )
        leg = bpy.context.active_object
        leg.name = "TableLeg"
        leg.data.materials.append(leg_mat)


# ── Mouse Model ──────────────────────────────────────────────────────────────
def build_mouse():
    """Build a more realistic mouse with body, head, ears, and tail."""
    mouse_mat = make_mouse_material()

    # Body — elongated ellipsoid
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.12, segments=24, ring_count=12,
        location=(0, 0, 0.10),
    )
    body = bpy.context.active_object
    body.name = "Mouse"
    body.scale = (1.8, 1.0, 0.7)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    body.data.materials.append(mouse_mat)

    # Head — smaller sphere at front
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.08, segments=20, ring_count=10,
        location=(0.20, 0, 0.12),
    )
    head = bpy.context.active_object
    head.name = "MouseHead"
    head.scale = (1.3, 1.0, 0.9)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    head.data.materials.append(mouse_mat)
    head.parent = body

    # Snout — tiny elongated sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.035, segments=12, ring_count=8,
        location=(0.30, 0, 0.11),
    )
    snout = bpy.context.active_object
    snout.name = "MouseSnout"
    snout.scale = (1.5, 0.8, 0.7)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    snout.data.materials.append(
        make_simple_material("SnoutMat", (0.25, 0.18, 0.14, 1.0), 0.6)
    )
    snout.parent = body

    # Nose tip
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.015, segments=8, ring_count=6,
        location=(0.34, 0, 0.115),
    )
    nose = bpy.context.active_object
    nose.name = "MouseNose"
    bpy.ops.object.shade_smooth()
    nose.data.materials.append(
        make_simple_material("NoseMat", (0.15, 0.08, 0.08, 1.0), 0.3)
    )
    nose.parent = body

    # Eyes
    ear_mat = make_simple_material("EarMat", (0.30, 0.22, 0.18, 1.0), 0.5)
    eye_mat = make_simple_material("EyeMat", (0.02, 0.02, 0.02, 1.0), 0.1)
    for side in [1, -1]:
        # Ears — flattened spheres
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.04, segments=12, ring_count=8,
            location=(0.22, side * 0.06, 0.20),
        )
        ear = bpy.context.active_object
        ear.name = f"MouseEar_{'L' if side > 0 else 'R'}"
        ear.scale = (0.6, 1.0, 1.3)
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.shade_smooth()
        ear.data.materials.append(ear_mat)
        ear.parent = body

        # Eyes — small dark spheres
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.012, segments=8, ring_count=6,
            location=(0.26, side * 0.04, 0.15),
        )
        eye = bpy.context.active_object
        eye.name = f"MouseEye_{'L' if side > 0 else 'R'}"
        bpy.ops.object.shade_smooth()
        eye.data.materials.append(eye_mat)
        eye.parent = body

    # Tail — bezier curve
    bpy.ops.curve.primitive_bezier_curve_add(location=(-0.20, 0, 0.08))
    tail = bpy.context.active_object
    tail.name = "MouseTail"
    tail.data.bevel_depth = 0.008
    tail.data.bevel_resolution = 4

    # Shape the tail curve
    points = tail.data.splines[0].bezier_points
    points[0].co = (0, 0, 0)
    points[0].handle_right = (0.1, 0, -0.02)
    points[0].handle_left = (-0.05, 0, 0.02)
    points[1].co = (-0.25, 0, 0.05)
    points[1].handle_left = (-0.15, 0, -0.02)
    points[1].handle_right = (-0.35, 0, 0.10)

    tail_mat = make_simple_material("TailMat", (0.30, 0.22, 0.18, 1.0), 0.5)
    tail.data.materials.append(tail_mat)
    tail.parent = body

    # Move to start position
    body.location = (-1.5, 0, 0.10)

    return body


def animate_mouse(mouse):
    """Walk from left to right with slight weaving for realism."""
    mouse.location = (-1.5, 0, 0.10)
    mouse.rotation_euler = (0, 0, 0)
    mouse.keyframe_insert(data_path="location", frame=1)
    mouse.keyframe_insert(data_path="rotation_euler", frame=1)

    # Slight turn as it weaves
    mouse.location = (-0.5, -0.25, 0.10)
    mouse.rotation_euler = (0, 0, math.radians(-8))
    mouse.keyframe_insert(data_path="location", frame=NUM_FRAMES // 3)
    mouse.keyframe_insert(data_path="rotation_euler", frame=NUM_FRAMES // 3)

    mouse.location = (0.5, -0.1, 0.10)
    mouse.rotation_euler = (0, 0, math.radians(5))
    mouse.keyframe_insert(data_path="location", frame=2 * NUM_FRAMES // 3)
    mouse.keyframe_insert(data_path="rotation_euler", frame=2 * NUM_FRAMES // 3)

    mouse.location = (1.5, 0, 0.10)
    mouse.rotation_euler = (0, 0, 0)
    mouse.keyframe_insert(data_path="location", frame=NUM_FRAMES)
    mouse.keyframe_insert(data_path="rotation_euler", frame=NUM_FRAMES)


# ── Camera & Lighting ─────────────────────────────────────────────────────────
def setup_camera():
    bpy.ops.object.camera_add(location=(0, -6, 5))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(55), 0, 0)
    cam.data.lens = 35  # slightly wider for kitchen feel
    bpy.context.scene.camera = cam
    return cam


def setup_lighting():
    # Main overhead area light (warm kitchen ceiling light)
    bpy.ops.object.light_add(type="AREA", location=(0, 1, 5.5))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 400
    key.data.size = 3.0
    key.data.color = (1.0, 0.95, 0.85)  # warm white

    # Under-cabinet light strip (illuminates counter/floor area)
    bpy.ops.object.light_add(type="AREA", location=(0, 4.2, 1.2))
    cab_light = bpy.context.active_object
    cab_light.name = "CabinetLight"
    cab_light.data.energy = 80
    cab_light.data.size = 3.0
    cab_light.rotation_euler = (math.radians(90), 0, 0)
    cab_light.data.color = (1.0, 0.97, 0.90)

    # Cool fill from the side (simulates window)
    bpy.ops.object.light_add(type="AREA", location=(-4, -2, 3.5))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 150
    fill.data.size = 2.5
    fill.rotation_euler = (math.radians(45), math.radians(45), 0)
    fill.data.color = (0.85, 0.90, 1.0)  # cool daylight tint

    # Subtle back-rim light for depth
    bpy.ops.object.light_add(type="POINT", location=(2, 5, 4))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 50
    rim.data.color = (1.0, 0.92, 0.80)


def setup_world():
    """Set up world background with warm ambient."""
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    bg = next(n for n in tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.05, 0.04, 0.035, 1.0)  # very dim warm ambient
    bg.inputs["Strength"].default_value = 0.3


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64  # decent quality, reasonable speed
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.image_settings.file_format = "PNG"
    scene.frame_start = 1
    scene.frame_end = NUM_FRAMES
    scene.render.fps = 30

    # Film settings
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Contrast"


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
            return None
        xs.append(ndc.x * RENDER_WIDTH)
        ys.append((1.0 - ndc.y) * RENDER_HEIGHT)

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
    print("\n=== Pest Detection Demo (Realistic) ===\n")

    clear_scene()
    build_kitchen()
    mouse = build_mouse()
    animate_mouse(mouse)
    camera = setup_camera()
    setup_lighting()
    setup_world()
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
