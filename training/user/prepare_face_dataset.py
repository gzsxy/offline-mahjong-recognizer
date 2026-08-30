#!/usr/bin/env python3
"""用户正面照片 → 模型辅助预标注（B 阶段 M7 用户域数据）。

流程（"模型预标注 + 人工核验"）：
1. 每张照片按部署同款几何切 1280x1280 重叠 25% 切片；
2. 用当前部署同源的 mj-v2 best.pt 在切片上推理（conf 0.15 偏召回）；
3. 坐标映射回原图，跨切片按类别 NMS；
4. 产出：1280 切片图 + YOLO 标签（微调数据集 mj-user-face-v1）+ 全图标注预览（人工核验用）。

注意：预标注质量由 previews/ 人工核验把关，错标会强化模型错误。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PHOTO_DIR = ROOT / "training/user/photos/20260829"
OUT = ROOT / "training/datasets/mj-user-face-v1"
PREVIEW_DIR = PHOTO_DIR / "previews"
MODEL = ROOT / "training/runs/mj-v2/weights/best.pt"

SLICE = 1280
OVERLAP = 0.25
CONF = 0.15
IOU_SLICE_NMS = 0.5
VAL_PHOTO = "face_mixed_b.jpg"
CLASS_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["back"]
)


def positions(total: int) -> list[int]:
    if total <= SLICE:
        return [0]
    stride = max(1, int(SLICE * (1.0 - OVERLAP)))
    last = total - SLICE
    result: list[int] = []
    position = 0
    while True:
        result.append(min(position, last))
        if position >= last:
            break
        position += stride
    return sorted(set(result))


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> list[int]:
    """跨切片全局按类别 NMS，返回保留索引。不同类别互不抑制。"""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.clip(xx1 - xx0, 0, None)
        h = np.clip(yy1 - yy0, 0, None)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        suppressed = (classes[rest] == classes[i]) & (iou > iou_thr)
        order = rest[~suppressed]
    return keep


def main() -> None:
    global SLICE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tile-size", type=int, default=1280,
                        help="切片边长（需与后续训练/部署 imgsz 一致）")
    parser.add_argument("--out", default=str(OUT), help="输出数据集目录")
    args = parser.parse_args()
    SLICE = args.tile_size
    OUT = Path(args.out)

    photos = sorted(PHOTO_DIR.glob("face_*.jpg"))
    if not photos:
        raise SystemExit(f"no face photos under {PHOTO_DIR}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(exist_ok=True)

    model = YOLO(str(MODEL))
    stats: dict[str, int] = {}

    for photo in photos:
        image = Image.open(photo).convert("RGB")
        width, height = image.size
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []

        for oy in positions(height):
            for ox in positions(width):
                crop = image.crop((ox, oy, min(ox + SLICE, width), min(oy + SLICE, height)))
                if crop.size != (SLICE, SLICE):
                    crop = crop.resize((SLICE, SLICE), Image.Resampling.BICUBIC)
                result = model.predict(crop, imgsz=SLICE, conf=CONF, iou=0.7,
                                       device=args.device, verbose=False)[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                xyxy = result.boxes.xyxy.cpu().numpy()
                # 切片被缩放过则映射回切片坐标，再平移到全图
                sx = SLICE / crop.width
                sy = SLICE / crop.height
                xyxy[:, [0, 2]] *= sx
                xyxy[:, [1, 3]] *= sy
                xyxy[:, [0, 2]] += ox
                xyxy[:, [1, 3]] += oy
                all_boxes.append(xyxy)
                all_scores.append(result.boxes.conf.cpu().numpy())
                all_classes.append(result.boxes.cls.cpu().numpy().astype(int))

        if not all_boxes:
            stats[photo.name] = 0
            continue
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)
        keep = nms(boxes, scores, classes, IOU_SLICE_NMS)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        stats[photo.name] = len(boxes)

        split = "val" if photo.name == VAL_PHOTO else "train"
        stem = photo.stem
        for index, (oy, ox) in enumerate(
            (oy, ox) for oy in positions(height) for ox in positions(width)
        ):
            crop = image.crop((ox, oy, min(ox + SLICE, width), min(oy + SLICE, height)))
            if crop.size != (SLICE, SLICE):
                crop = crop.resize((SLICE, SLICE), Image.Resampling.BICUBIC)
            sx = SLICE / crop.width
            sy = SLICE / crop.height
            lines: list[str] = []
            for box, cls in zip(boxes, classes):
                x0, y0, x1, y1 = box
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                bw, bh = x1 - x0, y1 - y0
                visible = (
                    max(0, min(x1, ox + SLICE) - max(x0, ox))
                    * max(0, min(y1, oy + SLICE) - max(y0, oy))
                )
                if visible / max(bw * bh, 1e-6) < 0.45:
                    continue
                ncx = np.clip((cx - ox) * sx / SLICE, 0, 1)
                ncy = np.clip((cy - oy) * sy / SLICE, 0, 1)
                nbw = min(bw * sx / SLICE, 1)
                nbh = min(bh * sy / SLICE, 1)
                lines.append(f"{int(cls)} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
            if not lines:
                continue
            name = f"{stem}_s{index:03d}"
            crop.save(OUT / f"images/{split}/{name}.jpg", quality=92)
            (OUT / f"labels/{split}/{name}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

        # 全图标注预览（人工核验用）
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        for box, cls, score in zip(boxes, classes, scores):
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline=(220, 30, 30), width=3)
            draw.text((x0 + 2, max(0, y0 - 14)),
                      f"{CLASS_NAMES[int(cls)]} {score:.2f}", fill=(220, 30, 30))
        preview.save(PREVIEW_DIR / f"{stem}_preview.jpg", quality=88)

    (OUT / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(OUT),
                "train": "images/train",
                "val": "images/val",
                "nc": len(CLASS_NAMES),
                "names": {i: n for i, n in enumerate(CLASS_NAMES)},
            },
            sort_keys=False,
            allow_unicode=False,
        )
    )
    for name, count in sorted(stats.items()):
        print(f"{name}: {count} tiles")
    print(f"dataset: {OUT}  previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
