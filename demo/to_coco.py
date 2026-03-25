"""
Convert output/annotations.json → output/coco_annotations.json (COCO format).

Run:
    python demo/to_coco.py
"""

import json
import os
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")
ANNO_PATH = os.path.join(OUTPUT_DIR, "annotations.json")
COCO_PATH = os.path.join(OUTPUT_DIR, "coco_annotations.json")

CATEGORIES = [
    {"id": 1, "name": "mouse", "supercategory": "pest"},
    {"id": 2, "name": "rat", "supercategory": "pest"},
    {"id": 3, "name": "cockroach", "supercategory": "pest"},
]
LABEL_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def main():
    with open(ANNO_PATH) as f:
        raw = json.load(f)

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for entry in raw:
        frame = entry["frame"]
        file_path = entry["file"]

        if not os.path.exists(file_path):
            print(f"  Skipping frame {frame} (file missing)")
            continue

        with Image.open(file_path) as img:
            w, h = img.size

        coco_images.append({
            "id": frame,
            "file_name": os.path.basename(file_path),
            "width": w,
            "height": h,
        })

        for pest in entry["pests"]:
            if pest is None:
                continue
            b = pest["bbox"]
            if b["width"] <= 0 or b["height"] <= 0:
                continue

            coco_annotations.append({
                "id": ann_id,
                "image_id": frame,
                "category_id": LABEL_TO_ID.get(pest["label"], 1),
                "bbox": [b["x_min"], b["y_min"], b["width"], b["height"]],  # [x, y, w, h]
                "area": b["width"] * b["height"],
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": "Pest Detection Synthetic Dataset", "version": "1.0"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    with open(COCO_PATH, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Saved {len(coco_images)} images, {len(coco_annotations)} annotations → {COCO_PATH}")


if __name__ == "__main__":
    main()
