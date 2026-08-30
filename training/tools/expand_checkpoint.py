#!/usr/bin/env python3
"""Expand the face checkpoint from 28 classes to the complete 43-class set.

The existing 27 suited-tile and back classifier weights are copied into the
new head. Honor classifiers start from the normal Detect initialization and
are then learned from the full public dataset.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "training/runs/mj-v2/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "training/runs/mj-full-v1/weights/expanded-init.pt"
CLASS_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["east", "south", "west", "north", "red", "green", "white"]
    + [f"flower{i}" for i in range(1, 5)]
    + [f"season{i}" for i in range(1, 5)]
    + ["back"]
)
OLD_CLASS_COUNT = 28
OLD_BACK_ID = 27
NEW_BACK_ID = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    source = YOLO(str(args.source))
    config = deepcopy(source.model.yaml)
    expanded = DetectionModel(config, ch=3, nc=len(CLASS_NAMES), verbose=False)

    old_state = source.model.state_dict()
    new_state = expanded.state_dict()
    copied = 0
    for key, value in old_state.items():
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key] = value
            copied += 1
    expanded.load_state_dict(new_state)

    old_head = source.model.model[-1]
    new_head = expanded.model[-1]
    for old_branch, new_branch in zip(old_head.cv3, new_head.cv3):
        old_classifier = old_branch[-1]
        new_classifier = new_branch[-1]
        with torch.no_grad():
            new_classifier.weight[:OLD_CLASS_COUNT].copy_(old_classifier.weight)
            new_classifier.bias[:OLD_CLASS_COUNT].copy_(old_classifier.bias)
            new_classifier.weight[NEW_BACK_ID].copy_(old_classifier.weight[OLD_BACK_ID])
            new_classifier.bias[NEW_BACK_ID].copy_(old_classifier.bias[OLD_BACK_ID])

    expanded.names = {index: name for index, name in enumerate(CLASS_NAMES)}
    source.model = expanded
    # Ultralytics reloads the EMA copy when it is present in a checkpoint.
    source.ckpt["ema"] = deepcopy(expanded).half()
    source.save(args.out)
    print(f"output: {args.out.resolve()}")
    print(f"class count: {len(CLASS_NAMES)}, copied tensors: {copied}")


if __name__ == "__main__":
    main()
