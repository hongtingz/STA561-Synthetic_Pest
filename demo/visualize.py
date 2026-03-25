"""
Visualize bounding boxes on rendered frames.
Reads output/annotations.json and draws green boxes on each frame,
saving results to output/viz/.

Run:
    python demo/visualize.py
"""

import json
import os
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")
ANNO_PATH = os.path.join(OUTPUT_DIR, "annotations.json")
VIZ_DIR = os.path.join(OUTPUT_DIR, "viz")

os.makedirs(VIZ_DIR, exist_ok=True)

LABEL_COLORS = {
    "mouse": (0, 220, 60),
    "rat": (220, 60, 0),
    "cockroach": (200, 180, 0),
}
DEFAULT_COLOR = (255, 255, 255)


def draw_frame(anno):
    img = Image.open(anno["file"]).convert("RGB")
    draw = ImageDraw.Draw(img)

    for pest in anno["pests"]:
        if pest is None:
            continue
        label = pest["label"]
        b = pest["bbox"]
        if b["width"] <= 0 or b["height"] <= 0:
            continue
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)

        # Bounding box rectangle
        draw.rectangle(
            [(b["x_min"], b["y_min"]), (b["x_max"], b["y_max"])],
            outline=color,
            width=2,
        )

        # Label background + text
        text = f"{label}"
        tx, ty = b["x_min"], max(0, b["y_min"] - 16)
        draw.rectangle([(tx, ty), (tx + len(text) * 7 + 4, ty + 14)], fill=color)
        draw.text((tx + 2, ty + 1), text, fill=(0, 0, 0))

    frame_name = f"frame_{anno['frame']:04d}.png"
    out_path = os.path.join(VIZ_DIR, frame_name)
    img.save(out_path)
    return out_path


def main():
    if not os.path.exists(ANNO_PATH):
        print(f"Annotations not found: {ANNO_PATH}")
        print("Run render_demo.py in Blender first.")
        return

    with open(ANNO_PATH) as f:
        annotations = json.load(f)

    print(f"Visualizing {len(annotations)} frames → {VIZ_DIR}")
    for anno in annotations:
        if not os.path.exists(anno["file"]):
            print(f"  Skipping frame {anno['frame']} (file missing)")
            continue
        out = draw_frame(anno)
        if anno["frame"] % 10 == 0 or anno["frame"] == 1:
            print(f"  Saved {out}")

    print(f"\nDone! Open {VIZ_DIR} to inspect results.")


if __name__ == "__main__":
    main()
