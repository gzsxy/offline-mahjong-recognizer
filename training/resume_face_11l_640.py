#!/usr/bin/env python3
"""从断点恢复正面 11l 预训（跨机器安全）。

last.pt 旁边的 args.yaml 记录着旧机器的 device（mps）与数据集绝对路径，
本脚本在 resume 前把它们改写为本机实际值，再交给 ultralytics 续训。
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "training/runs/mj-face-11l-640"
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

    args = yaml.safe_load(ARGS.read_text())
    args["device"] = pick_device()
    args["data"] = str(ROOT / "training/datasets/mj-v2/data.yaml")
    ARGS.write_text(yaml.safe_dump(args, sort_keys=False))
    print(f"resume args: device={args['device']} data={args['data']}")

    model = YOLO(str(LAST))
    model.train(resume=True)


if __name__ == "__main__":
    main()
