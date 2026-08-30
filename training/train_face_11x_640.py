#!/usr/bin/env python3
"""1920 轮正面预训：YOLO11x COCO 迁移（全新架构，无法从 11l 迁移权重），mj-v2 28 类。

开发文档 4.1-C。后续 1920 微调以本 run best.pt 为起点。
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/runs/pretrained/yolo11x.pt"
DATA = ROOT / "training/datasets/mj-v2/data.yaml"
PROJECT = ROOT / "training/runs"
NAME = "mj-face-11x-640"


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    if torch.cuda.is_available():
        device = 0  # DGX Spark / L40 等 CUDA 设备
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
        workers=8,
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
        fliplr=0.0,  # 麻将牌面有方向性
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
