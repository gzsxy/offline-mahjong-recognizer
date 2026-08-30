#!/usr/bin/env python3
"""Prepare labeled 640x640 crops from the user's blue/green back photos."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "training/user/photos/20260829"
DEFAULT_OUTPUT = ROOT / "training/datasets/mj-user-back-v1"
BACK_SPRITES_DIR = ROOT / "training/assets/back-sprites"
TILE_SIZE = 640
OVERLAP = 0.25
GRID_COLUMNS = 12
GRID_ROWS = 10
BACK_ID = 27
CLASS_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["back"]
)

# 四个角点是每张照片中 12x10 牌背网格的外角，按可见牌缝人工核对。
# 文件名对应 training/user/photos/20260829/ 下的作者实拍（该目录不入库），
# 使用你自己的照片时替换文件名与角点即可。
BACK_PHOTOS = {
    "back_blue_grid_a.jpg": ((428, 0), (1421, 36), (1686, 1258), (256, 1172)),
    "back_green_grid_a.jpg": ((413, 73), (1343, 75), (1703, 1206), (292, 1198)),
    "back_blue_grid_b.jpg": ((377, 0), (1388, 18), (1586, 1166), (198, 1164)),
    "back_green_grid_b.jpg": ((432, 45), (1398, 64), (1703, 1182), (299, 1199)),
}


def positions(total: int) -> list[int]:
    if total <= TILE_SIZE:
        return [0]
    stride = max(1, int(TILE_SIZE * (1.0 - OVERLAP)))
    last = total - TILE_SIZE
    result: list[int] = []
    position = 0
    while True:
        result.append(min(position, last))
        if position >= last:
            break
        position += stride
    return sorted(set(result))


def grid_boxes(image: Image.Image, corners: tuple[tuple[int, int], ...]) -> list[tuple[float, float, float, float]]:
    width, height = image.size
    source = np.float32(corners)
    target = np.float32(
        [[0, 0], [GRID_COLUMNS * 100, 0], [GRID_COLUMNS * 100, GRID_ROWS * 100], [0, GRID_ROWS * 100]]
    )
    inverse = np.linalg.inv(cv2.getPerspectiveTransform(source, target))
    boxes: list[tuple[float, float, float, float]] = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            # Leave a small margin around seams so labels do not include the gap.
            u0, v0 = column * 100 + 3, row * 100 + 3
            u1, v1 = (column + 1) * 100 - 3, (row + 1) * 100 - 3
            points = np.float32([[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]])
            projected = cv2.perspectiveTransform(points, inverse)[0]
            left = float(np.clip(projected[:, 0].min(), 0, width))
            top = float(np.clip(projected[:, 1].min(), 0, height))
            right = float(np.clip(projected[:, 0].max(), 0, width))
            bottom = float(np.clip(projected[:, 1].max(), 0, height))
            if right - left >= 10 and bottom - top >= 10:
                boxes.append((left, top, right, bottom))
    return boxes


def extract_back_sprites(
    image: Image.Image,
    corners: tuple[tuple[int, int], ...],
    source_stem: str,
) -> None:
    """Save perspective-corrected user back textures for scene synthesis."""
    source = np.float32(corners)
    target = np.float32(
        [[0, 0], [GRID_COLUMNS * 100, 0], [GRID_COLUMNS * 100, GRID_ROWS * 100], [0, GRID_ROWS * 100]]
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(np.asarray(image), matrix, (GRID_COLUMNS * 100, GRID_ROWS * 100))
    BACK_SPRITES_DIR.mkdir(parents=True, exist_ok=True)
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            crop = warped[row * 100 + 4:(row + 1) * 100 - 4, column * 100 + 4:(column + 1) * 100 - 4]
            Image.fromarray(crop, "RGB").save(
                BACK_SPRITES_DIR / f"user_{source_stem}_{row:02d}_{column:02d}.png"
            )


def crop_with_labels(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    offset_x: int,
    offset_y: int,
) -> tuple[Image.Image, list[str]]:
    right = min(offset_x + TILE_SIZE, image.width)
    bottom = min(offset_y + TILE_SIZE, image.height)
    crop_width = right - offset_x
    crop_height = bottom - offset_y
    crop = image.crop((offset_x, offset_y, right, bottom)).resize(
        (TILE_SIZE, TILE_SIZE), Image.Resampling.BICUBIC
    )
    scale_x = TILE_SIZE / crop_width
    scale_y = TILE_SIZE / crop_height
    labels: list[str] = []
    for left, top, box_right, box_bottom in boxes:
        visible_left = max(left, offset_x)
        visible_top = max(top, offset_y)
        visible_right = min(box_right, right)
        visible_bottom = min(box_bottom, bottom)
        if visible_right <= visible_left or visible_bottom <= visible_top:
            continue
        full_area = (box_right - left) * (box_bottom - top)
        visible_area = (visible_right - visible_left) * (visible_bottom - visible_top)
        if visible_area / full_area < 0.45:
            continue
        x0 = (visible_left - offset_x) * scale_x
        y0 = (visible_top - offset_y) * scale_y
        x1 = (visible_right - offset_x) * scale_x
        y1 = (visible_bottom - offset_y) * scale_y
        cx = (x0 + x1) / 2 / TILE_SIZE
        cy = (y0 + y1) / 2 / TILE_SIZE
        width = (x1 - x0) / TILE_SIZE
        height = (y1 - y0) / TILE_SIZE
        labels.append(f"{BACK_ID} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return crop, labels


def augment(image: Image.Image, rng: random.Random, index: int) -> Image.Image:
    result = image
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.75, 1.25))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.80, 1.20))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.85, 1.15))
    if index % 3 == 0:
        result = result.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if index % 4 == 0:
        pixels = np.asarray(result, dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(1.0, 5.0), pixels.shape)
        result = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8), "RGB")
    return result


def write_sample(image: Image.Image, labels: list[str], image_path: Path, label_path: Path) -> None:
    image.save(image_path, quality=92)
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.variants < 0:
        raise SystemExit("--variants must be non-negative")

    if args.out.exists():
        shutil.rmtree(args.out)
    train_image_dir = args.out / "images/train"
    val_image_dir = args.out / "images/val"
    train_label_dir = args.out / "labels/train"
    val_label_dir = args.out / "labels/val"
    for directory in (train_image_dir, val_image_dir, train_label_dir, val_label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    source_names = list(BACK_PHOTOS)
    val_names = {source_names[-1]}
    sample_count = {"train": 0, "val": 0}
    label_count = {"train": 0, "val": 0}

    for source_name, corners in BACK_PHOTOS.items():
        source_path = args.source / source_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        image = Image.open(source_path).convert("RGB")
        boxes = grid_boxes(image, corners)
        if len(boxes) != GRID_COLUMNS * GRID_ROWS:
            raise RuntimeError(f"{source_name}: expected 120 boxes, got {len(boxes)}")
        extract_back_sprites(image, corners, source_path.stem)
        split = "val" if source_name in val_names else "train"
        image_dir = val_image_dir if split == "val" else train_image_dir
        label_dir = val_label_dir if split == "val" else train_label_dir
        index = 0
        for offset_y in positions(image.height):
            for offset_x in positions(image.width):
                crop, labels = crop_with_labels(image, boxes, offset_x, offset_y)
                if not labels:
                    continue
                stem = f"{source_path.stem}_{index:03d}"
                write_sample(
                    crop,
                    labels,
                    image_dir / f"{stem}.jpg",
                    label_dir / f"{stem}.txt",
                )
                sample_count[split] += 1
                label_count[split] += len(labels)
                index += 1
                for variant in range(args.variants):
                    write_sample(
                        augment(crop.copy(), rng, variant),
                        labels,
                        image_dir / f"{stem}_aug{variant:02d}.jpg",
                        label_dir / f"{stem}_aug{variant:02d}.txt",
                    )
                    sample_count[split] += 1
                    label_count[split] += len(labels)

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False))
    print(f"output: {args.out.resolve()}")
    print(f"train samples: {sample_count['train']}, back labels: {label_count['train']}")
    print(f"val samples: {sample_count['val']}, back labels: {label_count['val']}")


if __name__ == "__main__":
    main()
