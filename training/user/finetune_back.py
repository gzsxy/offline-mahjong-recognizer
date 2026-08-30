#!/usr/bin/env python3
"""Adapt the verified back detector to rotated/scattered user textures."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
INITIAL = ROOT / "training/runs/mj-user-v1/weights/best.pt"
DATA = ROOT / "training/user/back-only.yaml"
PROJECT = ROOT / "training/runs"


def main() -> None:
    if not INITIAL.exists():
        raise FileNotFoundError(INITIAL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(INITIAL))
    result = model.train(
        data=str(DATA),
        epochs=4,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project=str(PROJECT),
        name="mj-user-v3",
        exist_ok=True,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.1,
        warmup_epochs=0.5,
        patience=4,
        amp=False,
        cache=False,
        close_mosaic=0,
        mosaic=0.1,
        mixup=0.0,
        degrees=8.0,
        translate=0.05,
        scale=0.25,
        perspective=0.001,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.35,
        seed=43,
        deterministic=True,
        plots=True,
    )
    print(f"training output: {result.save_dir}")


if __name__ == "__main__":
    main()
