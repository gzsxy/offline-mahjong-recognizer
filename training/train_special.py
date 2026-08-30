#!/usr/bin/env python3
"""Train a detector dedicated to honor and flower/season tiles."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/runs/mj-full-v1/weights/best.pt"
DATA = ROOT / "training/datasets/mj-special-v1/data.yaml"
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
        epochs=10,
        imgsz=512,
        batch=16,
        device=device,
        workers=4,
        project=str(PROJECT),
        name="mj-special-v1",
        exist_ok=False,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        warmup_epochs=2.0,
        patience=8,
        amp=True,
        cache=False,
        close_mosaic=3,
        mosaic=0.5,
        mixup=0.0,
        degrees=8.0,
        translate=0.08,
        scale=0.35,
        perspective=0.0005,
        fliplr=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        seed=47,
        deterministic=True,
        val=True,
        plots=True,
    )
    print(f"training output: {result.save_dir}")


if __name__ == "__main__":
    main()
