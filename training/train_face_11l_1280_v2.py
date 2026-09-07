#!/usr/bin/env python3
"""v2 方案 A2：正面 11l @1280 长程重训（保底冠军 + 11l 天花板对照）。

起始 = 640 预训 best（e55）；数据 = mj-face-1280-mix-v2（新场景合成 16k + v1 合成
2.5k + 用户切片 + mj-v2 30% 回放）。较 v1（12ep/b8/lr0 5e-4）显著加长加强。
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/runs/mj-face-11l-640/weights/best.pt"
DATA = ROOT / "training/datasets/mj-face-1280-mix-v2/data.yaml"
PROJECT = ROOT / "training/runs"
NAME = "mj-face-11l-1280-v2"

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
        epochs=48,
        imgsz=1280,
        batch=16,
        device=device,
        workers=8,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        warmup_epochs=3.0,
        patience=12,
        amp=True,
        cache=False,
        close_mosaic=12,
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
