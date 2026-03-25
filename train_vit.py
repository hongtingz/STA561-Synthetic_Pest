"""
ViT-based pest detector — fine-tunes google/vit-base-patch16-224 on
synthetic crops extracted from the Blender-rendered frames.

Approach:
  - Positive samples: mouse bounding-box crops from COCO annotations
  - Negative samples: random background crops from the same frames
  - Binary classifier (pest / no-pest)
  - Reports true detection rate and false positive rate on a held-out val split

Run:
    python train_vit.py
"""

import json
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_PATH = os.path.join(PROJECT_DIR, "output", "coco_annotations.json")
FRAMES_DIR = os.path.join(PROJECT_DIR, "output", "frames")

MODEL_NAME = "google/vit-base-patch16-224"
CROP_SIZE = 224          # ViT input size
NEG_PER_FRAME = 3        # background crops per frame
VAL_SPLIT = 0.2          # fraction of images held out for validation
EPOCHS = 10
BATCH_SIZE = 8
LR = 2e-5
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────
def build_samples(coco_path, frames_dir, neg_per_frame):
    """Return list of (PIL.Image crop, label) — 1=pest, 0=background."""
    with open(coco_path) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: os.path.join(frames_dir, img["file_name"])
                  for img in coco["images"]}
    id_to_anns = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)

    samples = []
    for img_id, file_path in id_to_file.items():
        if not os.path.exists(file_path):
            continue
        img = Image.open(file_path).convert("RGB")
        W, H = img.size

        # Positive crops
        for ann in id_to_anns[img_id]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            crop = img.crop((x, y, x + w, y + h)).resize((CROP_SIZE, CROP_SIZE))
            samples.append((crop, 1))

        # Negative crops (random regions that don't overlap any bbox)
        ann_boxes = [(a["bbox"][0], a["bbox"][1],
                      a["bbox"][0] + a["bbox"][2],
                      a["bbox"][1] + a["bbox"][3])
                     for a in id_to_anns[img_id]]
        attempts, added = 0, 0
        while added < neg_per_frame and attempts < 50:
            attempts += 1
            rx = random.randint(0, max(0, W - CROP_SIZE))
            ry = random.randint(0, max(0, H - CROP_SIZE))
            rx2, ry2 = rx + CROP_SIZE, ry + CROP_SIZE
            overlap = any(
                rx < bx2 and rx2 > bx1 and ry < by2 and ry2 > by1
                for bx1, by1, bx2, by2 in ann_boxes
            )
            if not overlap:
                crop = img.crop((rx, ry, rx2, ry2))
                samples.append((crop, 0))
                added += 1

    return samples


class CropDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("Loading COCO annotations and building crop dataset...")
    all_samples = build_samples(COCO_PATH, FRAMES_DIR, NEG_PER_FRAME)
    random.shuffle(all_samples)
    print(f"  Total crops: {len(all_samples)}  "
          f"(pos={sum(s[1]==1 for s in all_samples)}, "
          f"neg={sum(s[1]==0 for s in all_samples)})")

    split = int(len(all_samples) * (1 - VAL_SPLIT))
    train_samples, val_samples = all_samples[:split], all_samples[split:]

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    train_ds = CropDataset(train_samples, processor)
    val_ds = CropDataset(val_samples, processor)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Loading pretrained ViT: {MODEL_NAME}")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for pixel_values, labels in train_dl:
            pixel_values, labels = pixel_values.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        # Validation
        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for pixel_values, labels in val_dl:
                pixel_values, labels = pixel_values.to(DEVICE), labels.to(DEVICE)
                preds = model(pixel_values=pixel_values).logits.argmax(dim=1)
                for p, g in zip(preds, labels):
                    if g == 1 and p == 1: tp += 1
                    elif g == 0 and p == 1: fp += 1
                    elif g == 1 and p == 0: fn += 1
                    else: tn += 1

        tdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # true detection rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false positive rate
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"TDR={tdr:.1%}  FPR={fpr:.1%}")

    # Final check against project targets
    print("\n=== Final Evaluation ===")
    print(f"  True Detection Rate : {tdr:.1%}  (target ≥80%)")
    print(f"  False Positive Rate : {fpr:.1%}  (target <5%)")
    if tdr >= 0.80 and fpr < 0.05:
        print("  PASS")
    else:
        print("  FAIL — consider more synthetic frames or longer training")

    # Save model
    out_dir = os.path.join(PROJECT_DIR, "output", "vit_model")
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"\nModel saved → {out_dir}")


if __name__ == "__main__":
    train()
