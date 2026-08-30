#!/usr/bin/env python3
"""1920 轮牌背微调：YOLO11m 直接从现成 640 预训 best 起步（免二次预训），1920 混合集。"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/runs/mj-back-11m-640/weights/best.pt"
DATA = ROOT / "training/datasets/mj-back-1920-mix-v1/data.yaml"
PROJECT = ROOT / "training/runs"
NAME = "mj-back-11m-1920"


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = YOLO(str(MODEL))
    result = model.train(
        data=str(DATA),
        epochs=12,
        imgsz=1920,
        batch=8,
        device=device,
        workers=8,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=2.0,
        patience=5,
        amp=True,
        cache=False,
        close_mosaic=10,
        mosaic=0.75,
        mixup=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.35,
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
