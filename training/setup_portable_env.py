#!/usr/bin/env python3
"""迁移目标机（如 DGX Spark）解包后执行一次的本机化脚本。

做四件事：
1. 全部 training/datasets/*/data.yaml 的 path 改写为本机实际根目录；
2. 重建 mj-back-mix-v1 的符号链接（打包内是旧机器的绝对链接，在本机已断）；
3. 修正断点运行目录 args.yaml 里的数据集路径（device 由 resume 脚本处理）；
4. 打印 torch/CUDA/数据集状态，便于确认环境可用。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "training/datasets"
BACK_MIX = DATASETS / "mj-back-mix-v1"
BACK_MIX_SOURCES = ["mj-user-back-v1", "mj-user-back-synth-v1", "mj-backboost"]


def fix_data_yamls() -> None:
    for data_yaml in sorted(DATASETS.glob("*/data.yaml")):
        data = yaml.safe_load(data_yaml.read_text())
        correct = str(data_yaml.parent.resolve())
        if data.get("path") != correct:
            data["path"] = correct
            data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False))
            print(f"[yaml] {data_yaml.parent.name}: path -> {correct}")


def rebuild_back_mix() -> None:
    if not BACK_MIX.exists():
        print("[mix] mj-back-mix-v1 不在包内，跳过（需要时用 training/ 内合并脚本重建）")
        return
    for split in ("train", "val"):
        img_dir = BACK_MIX / f"images/{split}"
        lbl_dir = BACK_MIX / f"labels/{split}"
        for path in list(img_dir.iterdir()) + list(lbl_dir.iterdir()):
            if path.is_symlink():
                path.unlink()
        for src in BACK_MIX_SOURCES:
            src_root = DATASETS / src
            for img in sorted((src_root / f"images/{split}").iterdir()):
                lbl = src_root / f"labels/{split}" / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                (img_dir / f"{src}__{img.name}").symlink_to(img.resolve())
                (lbl_dir / f"{src}__{img.stem}.txt").symlink_to(lbl.resolve())
    print(f"[mix] mj-back-mix-v1 符号链接已重建（{BACK_MIX}）")


def fix_run_args() -> None:
    for args_yaml in sorted((ROOT / "training/runs").glob("*/args.yaml")):
        data = yaml.safe_load(args_yaml.read_text())
        data_path = data.get("data")
        if isinstance(data_path, str) and data_path.startswith("/"):
            name = Path(data_path).parent.name
            candidate = DATASETS / name / "data.yaml"
            if candidate.exists():
                data["data"] = str(candidate)
                args_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
                print(f"[args] {args_yaml.parent.name}: data -> {candidate}")


def report() -> None:
    print("=== 环境状态 ===")
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    for name in (
        "mj-v2", "mj-back-mix-v1", "mj-synth-1280-v1", "mj-back-synth-1280-v1",
        "mj-user-face-v1", "mj-user-back-1280-v1",
        "mj-user-back-v1", "mj-user-back-synth-v1", "mj-backboost",
    ):
        d = DATASETS / name / "images/train"
        n = len(list(d.iterdir())) if d.exists() else -1
        print(f"dataset {name}: {'缺失' if n < 0 else f'{n} train images'}")
    last = ROOT / "training/runs/mj-face-11l-640/weights/last.pt"
    print(f"断点 last.pt: {'存在' if last.exists() else '缺失'}")


if __name__ == "__main__":
    fix_data_yamls()
    rebuild_back_mix()
    fix_run_args()
    report()
