#!/usr/bin/env python3
"""B 阶段 M7 第一步：640 预训正面模型（YOLO11l，COCO 迁移，mj-v2 28 类）。

后续 1280 微调以此权重为起点（开发文档 4.1-B）。
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/runs/pretrained/yolo11l.pt"
DATA = ROOT / "training/datasets/mj-v2/data.yaml"
PROJECT = ROOT / "training/runs"
NAME = "mj-face-11l-640"


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    if torch.cuda.is_available():
        device = 0  # DGX Spark 等 CUDA 设备
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = YOLO(str(MODEL))
    result = model.train(
        data=str(DATA),
        epochs=60,
        imgsz=640,
        batch=8,
        device=device,
        workers=4,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        patience=15,
        amp=True,
        cache=False,
        close_mosaic=10,
        mosaic=0.75,
        mixup=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        perspective=0.0005,
        # 麻将牌面有方向性，不做水平翻转（与既有脚本一致）
        fliplr=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        seed=44,
        deterministic=True,
        plots=True,
    )
    print(f"training output: {result.save_dir}")


if __name__ == "__main__":
    main()
