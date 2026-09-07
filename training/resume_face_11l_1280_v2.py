#!/usr/bin/env python3
"""从断点恢复 v2 链 A2：正面 11l @1280 长程（2026-09-01 会话被杀后续训）。

中断点：e29 训练到 33%，last.pt 为 e28 完整落盘。本机路径完好，按
resume_face_11l_640.py 的防御模式：resume 前把 args.yaml 的 device/data
改写为本机实际值，再显式传给 train()。
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "training/runs/mj-face-11l-1280-v2"
ARGS = RUN / "args.yaml"
LAST = RUN / "weights/last.pt"


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    if not LAST.exists():
        raise FileNotFoundError(LAST)

    device = pick_device()
    data = str(ROOT / "training/datasets/mj-face-1280-mix-v2/data.yaml")

    # resume 时 ultralytics 用的是 last.pt 内嵌 train_args，会整体覆盖外部
    # 参数；仅 save_dir/device 等白名单键可再覆盖——这里三项都显式传入。
    args = yaml.safe_load(ARGS.read_text())
    args["device"] = device
    args["data"] = data
    ARGS.write_text(yaml.safe_dump(args, sort_keys=False))
    print(f"resume args: device={device} data={data} save_dir={RUN}")

    model = YOLO(str(LAST))
    model.train(resume=True, device=device, data=data, save_dir=str(RUN))


if __name__ == "__main__":
    main()
