#!/usr/bin/env python3
"""Crop real back-tile sprites from raw datasets into assets/back-sprites/."""
from pathlib import Path

from PIL import Image

TRAINING = Path(__file__).resolve().parents[2]
OUT = TRAINING / "training/assets/back-sprites"
OUT.mkdir(parents=True, exist_ok=True)


def find_image(imgs: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = imgs / (stem + ext)
        if p.exists():
            return p
    return None


def crop_from(split_root: Path, names, target_names, tag, center_frac=0.7):
    imgs, lbls = split_root / "images", split_root / "labels"
    if not lbls.exists():
        return 0
    n = 0
    for lf in sorted(lbls.glob("*.txt")):
        lines = [l for l in lf.read_text().strip().splitlines() if l.strip()]
        ids = [int(l.split()[0]) for l in lines]
        if tag == "ss6" and any(names[i] == "Mahjong_front" for i in ids):
            continue  # skip mixed images entirely
        img_path = find_image(imgs, lf.stem)
        if img_path is None:
            continue
        src = Image.open(img_path).convert("RGB")
        W, H = src.size
        for li, l in enumerate(lines):
            p = l.split()
            cid = int(p[0])
            if names[cid] not in target_names:
                continue
            cx, cy, bw, bh = map(float, p[1:5])
            x0, y0 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x1, y1 = (cx + bw / 2) * W, (cy + bh / 2) * H
            w, h = x1 - x0, y1 - y0
            mx, my = w * (1 - center_frac) / 2, h * (1 - center_frac) / 2
            box = (int(x0 + mx), int(y0 + my), int(x1 - mx), int(y1 - my))
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                continue
            src.crop(box).save(OUT / f"{tag}_{lf.stem}_{li}.png")
            n += 1
    return n


def main():
    n1 = 0
    for split in ("train", "valid", "test"):
        n1 += crop_from(
            TRAINING / f"training/datasets/raw/ss6ot_v1/{split}",
            ["Mahjong_back", "Mahjong_front"], {"Mahjong_back"}, "ss6")

    names37 = ['0m', '0p', '0s', '1m', '1p', '1s', '1z', '2m', '2p', '2s', '2z',
               '3m', '3p', '3s', '3z', '4m', '4p', '4s', '4z', '5m', '5p', '5s',
               '5z', '6m', '6p', '6s', '6z', '7m', '7p', '7s', '7z', '8m', '8p',
               '8s', '9m', '9p', '9s', 'back']
    n2 = 0
    for split in ("valid", "train", "test"):
        n2 += crop_from(
            TRAINING / f"training/datasets/raw/ma2nf_v5/{split}",
            names37, {"back"}, "ma")

    print(f"ss6 sprites: {n1}, ma2nf sprites: {n2}, total: {n1 + n2}")
    print(f"out: {OUT}")


if __name__ == "__main__":
    main()
