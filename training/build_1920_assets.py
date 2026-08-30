#!/usr/bin/env python3
"""构建 1920 轮训练混合集与导出校准集（符号链接，不复制图片）。

- mj-face-1920-mix-v1: mj-synth-1920-v1 + mj-user-face-1920-v1 + mj-v2 随机 30% 回放（seed 44）
- mj-back-1920-mix-v1: mj-back-synth-1920-v1 + mj-user-back-1920-v1
- calib-face-1920 / calib-back-1920: 导出 int8 用的校准集（合成 val + 用户实拍切片）
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "training/datasets"
SEED = 44
CLASSES = ([f"wan{i}" for i in range(1, 10)] + [f"tong{i}" for i in range(1, 10)]
           + [f"tiao{i}" for i in range(1, 10)] + ["back"])


def link_split(src_dirs: list[Path], fractions: list[float], out: Path, split: str,
               rng: random.Random | None) -> int:
    img_dir = out / f"images/{split}"
    lbl_dir = out / f"labels/{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src, frac in zip(src_dirs, fractions):
        src_img = src / f"images/{split}"
        if not src_img.exists():
            continue
        items = sorted(p for p in src_img.iterdir()
                       if (src / f"labels/{split}" / (p.stem + ".txt")).exists())
        if frac < 1.0 and rng is not None:
            items = rng.sample(items, k=int(len(items) * frac))
        for img in items:
            (img_dir / f"{src.name}__{img.name}").symlink_to(img.resolve())
            (lbl_dir / f"{src.name}__{img.stem}.txt").symlink_to(
                (src / f"labels/{split}" / (img.stem + ".txt")).resolve())
            n += 1
    return n


def build(name: str, train_parts: list[tuple[Path, float]],
          val_parts: list[tuple[Path, float]], seed: int | None) -> None:
    out = DATASETS / name
    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(seed) if seed is not None else None
    n_train = link_split([p for p, _ in train_parts], [f for _, f in train_parts], out, "train", rng)
    n_val = link_split([p for p, _ in val_parts], [f for _, f in val_parts], out, "val",
                       random.Random(seed + 1) if seed is not None else None)
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/train", "val": "images/val",
         "nc": 28, "names": {i: nm for i, nm in enumerate(CLASSES)}},
        sort_keys=False, allow_unicode=False))
    print(f"{name}: train {n_train}, val {n_val}")


def main() -> None:
    synth_face = DATASETS / "mj-synth-1920-v1"
    synth_back = DATASETS / "mj-back-synth-1920-v1"
    user_face = DATASETS / "mj-user-face-1920-v1"
    user_back = DATASETS / "mj-user-back-1920-v1"
    mjv2 = DATASETS / "mj-v2"
    for d in (synth_face, synth_back, user_face, user_back, mjv2):
        if not d.exists():
            raise SystemExit(f"缺少数据集：{d}")

    build("mj-face-1920-mix-v1",
          train_parts=[(synth_face, 1.0), (user_face, 1.0), (mjv2, 0.3)],
          val_parts=[(synth_face, 1.0), (user_face, 1.0), (mjv2, 0.3)], seed=SEED)
    build("mj-back-1920-mix-v1",
          train_parts=[(synth_back, 1.0), (user_back, 1.0)],
          val_parts=[(synth_back, 1.0), (user_back, 1.0)], seed=None)
    # 校准集：train 即 val，图片量小、域覆盖优先
    build("calib-face-1920",
          train_parts=[(synth_face, 1.0), (user_face, 1.0)],
          val_parts=[(synth_face, 1.0), (user_face, 1.0)], seed=None)
    build("calib-back-1920",
          train_parts=[(synth_back, 1.0), (user_back, 1.0)],
          val_parts=[(synth_back, 1.0), (user_back, 1.0)], seed=None)


if __name__ == "__main__":
    main()
