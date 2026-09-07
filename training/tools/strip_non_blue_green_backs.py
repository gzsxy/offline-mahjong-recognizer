#!/usr/bin/env python3
"""剔除非蓝/绿牌背标签（2026-09-08 需求：牌背限定蓝绿）。

对数据集每个 back 类标签框做蓝/绿像素占比检查（与 App 端
hasColoredBackAppearance 同口径），不达标的标签行删除——被删对象留在图中
成为负样本，教会模型"黑背/红背/杂色背不是牌背"。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/wang/zcode/majiang")
BACK_ID = 27


def blue_green_ratio(pil_img: Image.Image) -> float:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.int32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    spread = mx - mn
    colored = (mx > 0) & (spread > 25) & (spread * 5 > mx)
    blue_back = (b * 100 > r * 112) & (b * 100 > g * 103)
    green_back = (g * 100 > r * 112) & (g * 100 > b * 103)
    return float((colored & (blue_back | green_back)).mean())


def curate(dataset: Path, min_ratio: float, apply: bool) -> tuple[int, int]:
    removed = kept = 0
    for split in ("train", "val"):
        lbl_dir = dataset / f"labels/{split}"
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            img_path = dataset / f"images/{split}/{lbl.stem}.jpg"
            if not img_path.exists():
                continue
            image = Image.open(img_path)
            W, H = image.size
            lines = lbl.read_text().splitlines()
            out_lines = []
            changed = False
            for line in lines:
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) != BACK_ID:
                    out_lines.append(line)
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                x0 = max(0, int((cx - bw / 2) * W))
                y0 = max(0, int((cy - bh / 2) * H))
                x1 = min(W, int((cx + bw / 2) * W))
                y1 = min(H, int((cy + bh / 2) * H))
                if x1 - x0 < 8 or y1 - y0 < 8:
                    out_lines.append(line)
                    continue
                ratio = blue_green_ratio(image.crop((x0, y0, x1, y1)))
                if ratio >= min_ratio:
                    out_lines.append(line)
                    kept += 1
                else:
                    changed = True
                    removed += 1
            if changed and apply:
                lbl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return removed, kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=[
        "mj-backboost", "mj-user-back-synth-v1", "mj-user-back-v1", "mj-v2",
    ])
    parser.add_argument("--min-ratio", type=float, default=0.18)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    total_removed = 0
    for name in args.datasets:
        ds = ROOT / "training/datasets" / name
        if not ds.exists():
            print(f"{name}: 不存在，跳过")
            continue
        removed, kept = curate(ds, args.min_ratio, args.apply)
        total_removed += removed
        print(f"{name}: 剔除 {removed} 个非蓝绿背标签，保留 {kept} 个（apply={args.apply}）")
    print(f"合计剔除: {total_removed}")


if __name__ == "__main__":
    main()
