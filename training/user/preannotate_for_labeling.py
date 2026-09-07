#!/usr/bin/env python3
"""人工标注辅助：把新照片预标注成 YOLO 格式，供 labelImg 修正。

与部署口径对齐（conf 0.45、双模型、跨切片 IoU 0.30 + 包含率 0.60 去重），
保证"机器先标一遍、人只做修正"的量最小。输出：
  <out>/images/  照片本体（1280 切片，与训练分布一致）
  <out>/labels/  YOLO 标签
  <out>/classes.txt  labelImg 类别表

用法：
  training/venv/bin/python training/user/preannotate_for_labeling.py \
      --photos <照片目录或单张> --out training/datasets/<名字>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path("/Users/wang/zcode/majiang")
FACE = ROOT / "training/runs/mj-face-11l-1280/weights/best.pt"
BACK = ROOT / "training/runs/mj-back-11m-1280/weights/best.pt"
TILE = 1280
CONF = 0.45
CLASS_NAMES = ([f"wan{i}" for i in range(1, 10)] + [f"tong{i}" for i in range(1, 10)]
               + [f"tiao{i}" for i in range(1, 10)] + ["back"])
BACK_ID = 27


def positions(total: int) -> list[int]:
    if total <= TILE:
        return [0]
    stride = int(TILE * 0.75)
    last = total - TILE
    out, p = [], 0
    while True:
        out.append(min(p, last))
        if p >= last:
            break
        p += stride
    return sorted(set(out))


def detect(model, image):
    boxes, scores, classes = [], [], []
    for oy in positions(image.height):
        for ox in positions(image.width):
            crop = image.crop((ox, oy, min(ox + TILE, image.width), min(oy + TILE, image.height)))
            if crop.size != (TILE, TILE):
                crop = crop.resize((TILE, TILE), Image.BICUBIC)
            r = model.predict(crop, imgsz=TILE, conf=CONF, iou=0.7, device="cpu", verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            sx, sy = TILE / crop.width, TILE / crop.height
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


def class_nms(boxes, scores, classes, iou_thr=0.30, contain_thr=0.60):
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
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(ai + ar - inter, 1e-6)
        containment = inter / np.maximum(np.minimum(ai, ar), 1e-6)
        order = rest[~((classes[rest] == classes[i]) & ((iou > iou_thr) | (containment > contain_thr)))]
    return boxes[keep], scores[keep], classes[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", required=True, help="照片目录或单张照片路径")
    parser.add_argument("--out", required=True, help="输出数据集目录")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--previews", action="store_true", help="同时输出全图标注预览便于核对")
    args = parser.parse_args()

    src = Path(args.photos)
    photos = [src] if src.is_file() else sorted(p for p in src.glob("*.jpg"))
    if not photos:
        raise SystemExit(f"没有找到照片：{src}")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    (out / "previews").mkdir(parents=True)
    (out / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    face = YOLO(str(FACE))
    back = YOLO(str(BACK))

    for photo in photos:
        image = Image.open(photo).convert("RGB")
        fb, fs, fc = detect(face, image)
        bb, bs, bc = detect(back, image)
        fb, fs, fc = class_nms(fb, fs, fc)
        # 牌背只取 back 类；正面模型输出的 back 类丢弃（域分工与部署一致）
        bb, bs, bc = class_nms(bb[bc == BACK_ID], bs[bc == BACK_ID], bc[bc == BACK_ID])
        boxes = np.vstack([b for b in (fb, bb) if len(b)]) if (len(fb) or len(bb)) else np.zeros((0, 4))
        classes = np.concatenate([fc, bc]).astype(int) if (len(fc) or len(bc)) else np.zeros(0, dtype=int)

        W, H = image.size
        for oy in positions(H):
            for ox in positions(W):
                crop = image.crop((ox, oy, min(ox + TILE, W), min(oy + TILE, H)))
                if crop.size != (TILE, TILE):
                    crop = crop.resize((TILE, TILE), Image.BICUBIC)
                cw, ch = crop.size
                lines = []
                for (x0, y0, x1, y1), cls in zip(boxes, classes):
                    vx0, vy0 = max(x0, ox), max(y0, oy)
                    vx1, vy1 = min(x1, ox + TILE), min(y1, oy + TILE)
                    if vx1 <= vx0 or vy1 <= vy0:
                        continue
                    full = (x1 - x0) * (y1 - y0)
                    if (vx1 - vx0) * (vy1 - vy0) / max(full, 1e-6) < 0.45:
                        continue
                    cx = ((vx0 + vx1) / 2 - ox) / cw
                    cy = ((vy0 + vy1) / 2 - oy) / ch
                    bw = (vx1 - vx0) / cw
                    bh = (vy1 - vy0) / ch
                    lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stem = f"{photo.stem}_x{ox}_y{oy}"
                crop.save(out / f"images/{stem}.jpg", quality=92)
                (out / f"labels/{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if args.previews:
            pv = image.copy()
            dr = ImageDraw.Draw(pv)
            for (x0, y0, x1, y1), cls in zip(boxes, classes):
                color = (60, 160, 60) if cls != BACK_ID else (230, 170, 30)
                dr.rectangle([x0, y0, x1, y1], outline=color, width=3)
                dr.text((x0 + 2, max(0, y0 - 14)), CLASS_NAMES[int(cls)], fill=color)
            pv.save(out / f"previews/{photo.stem}_preview.jpg", quality=88)

        print(f"{photo.name}: {len(boxes)} 个预标注框")

    print(f"\n输出：{out}")
    print("labelImg 打开方式：labelImg " + str(out / "images") + " " +
          str(out / "classes.txt") + " " + str(out / "labels"))


if __name__ == "__main__":
    main()
