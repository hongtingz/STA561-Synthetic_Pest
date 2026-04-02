"""
Sliding-window pest detector using the fine-tuned ViT.

Uses multi-scale windows to handle pests of different sizes.
Each window is resized to 224x224 before classification (matching training).

Usage:
    python infer.py path/to/your/kitchen.jpg

Outputs:
    - Prints detections with confidence scores
    - Saves annotated image to output/infer_result.jpg
"""

import sys
import os
import torch
from PIL import Image, ImageDraw
from transformers import ViTForImageClassification, ViTImageProcessor

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "output", "vit_model")
OUT_PATH = os.path.join(PROJECT_DIR, "output", "infer_result.jpg")

# Window sizes to try — covers small pests (~60px) up to large ones (~200px)
WINDOW_SIZES = [60, 100, 150, 200]
STRIDE_RATIO = 0.3   # stride = 30% of window size
THRESHOLD = 0.9      # confidence to count as detection
NMS_IOU = 0.3        # suppress overlapping boxes above this IoU

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def iou(a, b):
    """Intersection over union of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections):
    """Non-maximum suppression — keep highest-score box when boxes overlap."""
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    for det in detections:
        x1, y1, x2, y2, score = det
        if all(iou((x1, y1, x2, y2), (k[0], k[1], k[2], k[3])) < NMS_IOU for k in kept):
            kept.append(det)
    return kept


def run(image_path):
    print(f"Loading model from {MODEL_DIR}")
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model = ViTForImageClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    print(f"Image size: {W}x{H}  |  device: {DEVICE}")
    print(f"Window sizes: {WINDOW_SIZES}  |  threshold: {THRESHOLD}")

    raw_detections = []  # (x1, y1, x2, y2, score)

    with torch.no_grad():
        for win in WINDOW_SIZES:
            stride = max(1, int(win * STRIDE_RATIO))
            for y in range(0, H - win + 1, stride):
                for x in range(0, W - win + 1, stride):
                    crop = img.crop((x, y, x + win, y + win))
                    # resize to 224x224 — matches how training crops were built
                    crop = crop.resize((224, 224))
                    inputs = processor(images=crop, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(DEVICE)
                    logits = model(pixel_values=pixel_values).logits
                    probs = torch.softmax(logits, dim=1)
                    pest_score = probs[0, 1].item()
                    if pest_score >= THRESHOLD:
                        raw_detections.append((x, y, x + win, y + win, pest_score))

    detections = nms(raw_detections)
    print(f"\nRaw detections: {len(raw_detections)}  →  after NMS: {len(detections)}")
    for x1, y1, x2, y2, score in detections:
        print(f"  bbox=({x1}, {y1}, {x2}, {y2})  confidence={score:.1%}")

    # Draw results
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, score in detections:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 4, y1 + 4), f"pest {score:.0%}", fill="red")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    img.save(OUT_PATH)
    print(f"\nAnnotated image saved → {OUT_PATH}")

    if not detections:
        print("No pests detected. Try lowering THRESHOLD or check the image.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py path/to/image.jpg")
        sys.exit(1)
    run(sys.argv[1])
