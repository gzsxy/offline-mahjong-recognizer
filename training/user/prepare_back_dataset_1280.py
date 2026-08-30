#!/usr/bin/env python3
"""用户 4 张 12x10 牌背网格图 → 1280x1280 切片数据集（牌背 11m 的 1280 微调用）。

复用 prepare_back_dataset 的网格角点与几何标注，仅改切片尺寸；牌背纹理精灵
已由原脚本提取过，这里不再重复提取。
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

import prepare_back_dataset as base

ROOT = base.ROOT
SOURCE_DIR = ROOT / "training/user/photos/20260829"
OUT = ROOT / "training/datasets/mj-user-back-1280-v1"
VARIANTS = 3
SPLIT_SEED = 42


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", type=int, default=1280,
                        help="切片边长（需与后续训练/部署 imgsz 一致）")
    parser.add_argument("--out", default=str(OUT), help="输出数据集目录")
    args = parser.parse_args()

    base.TILE_SIZE = args.tile_size
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SPLIT_SEED)
    photos = list(base.BACK_PHOTOS.items())
    val_name = photos[-1][0]  # 与原脚本一致：最后一张作 val
    counts = {"train": 0, "val": 0}

    for name, corners in photos:
        image = base.Image.open(SOURCE_DIR / name).convert("RGB")
        boxes = base.grid_boxes(image, corners)
        if len(boxes) != base.GRID_COLUMNS * base.GRID_ROWS:
            raise RuntimeError(f"{name}: expected 120 boxes, got {len(boxes)}")
        split = "val" if name == val_name else "train"
        index = 0
        for oy in base.positions(image.height):
            for ox in base.positions(image.width):
                crop, labels = base.crop_with_labels(image, boxes, ox, oy)
                if not labels:
                    continue
                stem = f"{Path(name).stem}_{index:03d}"
                base.write_sample(
                    crop, labels,
                    out_dir / f"images/{split}/{stem}.jpg",
                    out_dir / f"labels/{split}/{stem}.txt",
                )
                counts[split] += 1
                for v in range(VARIANTS):
                    base.write_sample(
                        base.augment(crop.copy(), rng, v), labels,
                        out_dir / f"images/{split}/{stem}_aug{v:02d}.jpg",
                        out_dir / f"labels/{split}/{stem}_aug{v:02d}.txt",
                    )
                    counts[split] += 1
                index += 1

    (out_dir / "data.yaml").write_text(yaml.safe_dump(
        {
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(base.CLASS_NAMES),
            "names": {i: n for i, n in enumerate(base.CLASS_NAMES)},
        },
        sort_keys=False, allow_unicode=False,
    ))
    print(f"out: {out_dir.resolve()}")
    print(f"tile: {args.tile_size}, train: {counts['train']}, val: {counts['val']}")


if __name__ == "__main__":
    main()
