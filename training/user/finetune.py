#!/usr/bin/env python3
"""Fine-tune the merged public-data model with user back photos."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "training/runs/mj-v2/weights/best.pt"
DATA = ROOT / "training/datasets/mj-user-finetune-v1/data.yaml"
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
        epochs=12,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project=str(PROJECT),
        name="mj-user-v2",
        exist_ok=True,
        pretrained=False,
        lr0=0.0005,
        lrf=0.05,
        warmup_epochs=1.0,
        patience=5,
        amp=False,
        cache=False,
        close_mosaic=0,
        mosaic=0.1,
        mixup=0.0,
        degrees=4.0,
        translate=0.05,
        scale=0.25,
        perspective=0.0005,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        seed=42,
        deterministic=True,
        plots=True,
    )
    print(f"training output: {result.save_dir}")


if __name__ == "__main__":
    main()
