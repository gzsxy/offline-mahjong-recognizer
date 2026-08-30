#!/usr/bin/env python3
"""Spot-check: render YOLO label boxes from dataset txt files onto images."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from generate import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "training/datasets/mj-synth-v1"
OUT = ROOT / "training/assets/preview"

def render(split, idx):
    name = f"mj_{split}_{idx:05d}"
    img = Image.open(DS / "images" / split / f"{name}.jpg").convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    w, h = img.size
    lines = (DS / "labels" / split / f"{name}.txt").read_text().strip().splitlines()
    for ln in lines:
        cid, cx, cy, bw, bh = ln.split()
        cid = int(cid); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
        color = (255, 60, 60) if cid == 27 else (255, 220, 40)
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
        d.text((x0 + 2, max(0, y0 - 15)), CLASS_NAMES[cid], fill=color, font=font)
    out = OUT / f"check_{split}_{idx}.jpg"
    img.save(out, quality=92)
    print(f"{name}: {len(lines)} boxes -> {out}")

if __name__ == "__main__":
    for split, idx in [("train", 7), ("train", 1234), ("val", 42)]:
        render(split, idx)
