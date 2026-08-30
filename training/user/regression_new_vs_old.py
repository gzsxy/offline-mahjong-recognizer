#!/usr/bin/env python3
"""B7 桌面回归：新 1280 双模型 vs 旧三模型部署口径，在 11 张用户实拍上对比。

- 旧 face：mj-v2 best.pt @640 切片 conf 0.45（现部署口径）
- 旧 back：mj-user-v1 @0.45 + mj-user-v3 @0.20（色彩校验无法离线复刻，用低阈并集近似）
- 新 face：mj-face-11l-1280 best.pt @1280 切片 conf 0.45
- 新 back：mj-back-11m-1280 best.pt @1280 切片 conf 0.45
- 已知硬真值：4 张牌背网格 = 120 张（12×10，角点核对过）
每张照片两个模型都跑，观察跨误检。输出对比表 + 3 张标注预览。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PHOTOS = sorted((ROOT / "training/user/photos/20260829").glob("*.jpg"))
OUT = ROOT / "training/user/regression_20260830"
OUT.mkdir(exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSES = ([f"wan{i}" for i in range(1, 10)] + [f"tong{i}" for i in range(1, 10)]
           + [f"tiao{i}" for i in range(1, 10)] + ["back"])
BACK_ID = 27
PREVIEW_FOR = {"face_mixed_c", "face_grid_rot_a", "back_blue_grid_a"}


def positions(total: int, tile: int) -> list[int]:
    if total <= tile:
        return [0]
    stride = max(1, int(tile * 0.75))
    last = total - tile
    result, position = [], 0
    while True:
        result.append(min(position, last))
        if position >= last:
            break
        position += stride
    return sorted(set(result))


def detect(model: YOLO, image: Image.Image, tile: int, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, scores, classes = [], [], []
    for oy in positions(image.height, tile):
        for ox in positions(image.width, tile):
            crop = image.crop((ox, oy, min(ox + tile, image.width), min(oy + tile, image.height)))
            if crop.size != (tile, tile):
                crop = crop.resize((tile, tile), Image.Resampling.BICUBIC)
            r = model.predict(crop, imgsz=tile, conf=conf, iou=0.7, device=DEVICE, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            sx, sy = tile / crop.width, tile / crop.height
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy
            xyxy[:, [0, 2]] += ox
            xyxy[:, [1, 3]] += oy
            boxes.append(xyxy)
            scores.append(r.boxes.conf.cpu().numpy())
            classes.append(r.boxes.cls.cpu().numpy().astype(int))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)


def class_nms(boxes, scores, classes, iou=0.3):
    if len(boxes) == 0:
        return boxes, scores, classes
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou_v = inter / np.maximum(a_i + a_r - inter, 1e-6)
        suppressed = (classes[rest] == classes[i]) & (iou_v > iou)
        order = rest[~suppressed]
    return boxes[keep], scores[keep], classes[keep]


def main() -> None:
    old_face = YOLO(str(ROOT / "training/runs/mj-v2/weights/best.pt"))
    old_back1 = YOLO(str(ROOT / "training/runs/mj-user-v1/weights/best.pt"))
    old_back2 = YOLO(str(ROOT / "training/runs/mj-user-v3/weights/best.pt"))
    new_face = YOLO(str(ROOT / "training/runs/mj-face-11l-1280/weights/best.pt"))
    new_back = YOLO(str(ROOT / "training/runs/mj-back-11m-1280/weights/best.pt"))

    header = f"{'photo':24s} {'旧face':>7s} {'新face':>7s} {'旧back':>7s} {'新back':>7s} | {'新模型互串':>8s}"
    print(header)
    print("-" * len(header))
    for photo in PHOTOS:
        image = Image.open(photo).convert("RGB")

        ob, os_, oc = detect(old_face, image, 640, 0.45)
        ob, os_, oc = class_nms(ob, os_, oc)
        old_face_n = int((oc != BACK_ID).sum())

        nb, ns, nc = detect(new_face, image, 1280, 0.45)
        nb, ns, nc = class_nms(nb, ns, nc)
        new_face_n = int((nc != BACK_ID).sum())
        new_face_back_fp = int((nc == BACK_ID).sum())

        b1, s1, _ = detect(old_back1, image, 640, 0.45)
        b2, s2, _ = detect(old_back2, image, 640, 0.20)
        ab = np.concatenate([b1, b2]); asc = np.concatenate([s1, s2])
        ab, asc, acl = class_nms(ab, asc, np.full(len(asc), BACK_ID))
        old_back_n = len(ab)

        bb, bs, bc = detect(new_back, image, 1280, 0.45)
        bb, bs, bc = class_nms(bb, bs, bc)
        new_back_n = int((bc == BACK_ID).sum())
        new_back_face_fp = len(bc) - new_back_n

        print(f"{photo.name:24s} {old_face_n:7d} {new_face_n:7d} {old_back_n:7d} {new_back_n:7d} | "
              f"face→back {new_face_back_fp:2d}, back→face {new_back_face_fp:2d}")

        if photo.stem in PREVIEW_FOR:
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            for box, score, cls in zip(nb, ns, nc):
                if cls == BACK_ID:
                    continue
                draw.rectangle([*box], outline=(30, 160, 60), width=3)
            for box, score in zip(bb, bs):
                draw.rectangle([*box], outline=(230, 170, 30), width=3)
            preview.save(OUT / f"{photo.stem}_new_preview.jpg", quality=88)

    print(f"\n预览: {OUT}")
    print("真值参考：4 张牌背网格 = 120 张；正面网格为同一副牌（约 120，无人工点数）")


if __name__ == "__main__":
    main()
