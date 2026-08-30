#!/usr/bin/env python3
"""Fine-tune the complete detector with flower/season oversampling."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/runs/mj-full-v1/weights/best.pt"
DATA = ROOT / "training/datasets/mj-full-balanced-v1/data.yaml"
PROJECT = ROOT / "training/runs"


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(MODEL))
    result = model.train(
        data=str(DATA),
        epochs=8,
        imgsz=512,
        batch=16,
        device=device,
        workers=4,
        project=str(PROJECT),
        name="mj-full-balanced-v1",
        exist_ok=False,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.00025,
        lrf=0.05,
        warmup_epochs=1.0,
        patience=8,
        amp=True,
        cache=False,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.25,
        perspective=0.0003,
        fliplr=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        seed=46,
        deterministic=True,
        val=False,
        plots=False,
    )
    print(f"training output: {result.save_dir}")


if __name__ == "__main__":
    main()
