#!/usr/bin/env python3
"""B 阶段 M7 第一步：640 预训牌背模型（YOLO11m，COCO 迁移，牌背混合集）。

数据 = 用户网格切片 + 用户纹理散放合成 + backboost 合成（mj-back-mix-v1）。
部署上负责牌背检测；牌面在该数据集中作为背景学习，抑制把正面误报为牌背。
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/runs/pretrained/yolo11m.pt"
DATA = ROOT / "training/datasets/mj-back-mix-v1/data.yaml"
PROJECT = ROOT / "training/runs"
NAME = "mj-back-11m-640"


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
        batch=12,
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
