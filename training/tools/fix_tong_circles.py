#!/usr/bin/env python3
"""修正用户正面预标注里的 二筒/四筒 混淆。

该用户牌具的画法：二筒 = 2 个大同心圆（直径约为牌面高度的 45%+），
四筒 = 4 个小圆（2x2，单个直径约为高度的 25%）。用 Hough 圆检测量
框内最大圆半径占牌高的比例即可无歧义区分：
  r_max / h > 0.20 → 二筒(tong2)
  检出 >=4 个小圆且 r_max / h < 0.17 → 四筒(tong4)
介于两者之间的不动（宁缺勿错）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "training/datasets/mj-user-face-v1"
TONG2, TONG4 = 10, 12  # class id


def classify(crop_bgr: np.ndarray, desqueeze: tuple[float, float] | None = None) -> int | None:
    """返回 10/12 的判定，None 表示无法确定。

    desqueeze: (x_factor, y_factor)——切片若经历过非等比缩放（如小图拉伸到方形切片），
    先按原纵横比缩回再检测，否则圆形变椭圆会让半径判定失真。
    """
    height, width = crop_bgr.shape[:2]
    if height < 40 or width < 40:
        return None
    if desqueeze is not None:
        crop_bgr = cv2.resize(
            crop_bgr,
            (max(8, int(width * desqueeze[0])), max(8, int(height * desqueeze[1]))),
        )
        height, width = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(height * 0.22),
        param1=120,
        param2=28,
        minRadius=max(6, int(height * 0.06)),
        maxRadius=int(height * 0.45),
    )
    if circles is None:
        return None
    # 同心圆多环会在同一圆心返回多个半径，先按圆心去重（每圆心只留最大半径）
    det = circles[0]
    order = np.argsort(det[:, 2])[::-1]
    kept_centers: list[tuple[float, float]] = []
    kept_radii: list[float] = []
    for idx in order:
        cx, cy, r = det[idx]
        if any((cx - kx) ** 2 + (cy - ky) ** 2 < (height * 0.25) ** 2
               for kx, ky in kept_centers):
            continue
        kept_centers.append((float(cx), float(cy)))
        kept_radii.append(float(r) / height)
    radii = np.sort(np.array(kept_radii))[::-1]
    # 按圆的数量判别（尺度无关，不依赖框的松紧）：
    #   二筒 = 2 个大同心圆；四筒 = 4 个小圆(2x2)
    big = radii[radii > 0.12]
    r_max = float(radii.max())
    if len(big) >= 4 and r_max < 0.30:
        return TONG4
    if len(big) == 2 and r_max > 0.28:
        return TONG2
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="不加则只报告不写回")
    parser.add_argument("--dataset", default=str(DATASET), help="目标数据集根目录")
    parser.add_argument("--desqueeze", default="",
                        help="切片经非等比缩放时按 '源宽,源高' 复原纵横比，如 1704,1279")
    parser.add_argument("--slice-size", type=int, default=1920,
                        help="与 --desqueeze 配套：切片边长")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    desqueeze = None
    if args.desqueeze:
        sw, sh = (float(v) for v in args.desqueeze.split(","))
        desqueeze = (sw / args.slice_size, sh / args.slice_size)

    fixed = kept = unsure = 0
    for label_path in sorted(dataset.glob("labels/*/*.txt")):
        image_path = label_path.parent.parent.parent / label_path.relative_to(
            label_path.parent.parent
        ).parents[1].joinpath("")  # placeholder, resolved below
        split = label_path.parent.name
        image_path = dataset / f"images/{split}/{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        lines = label_path.read_text().splitlines()
        changed = False
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) not in (TONG2, TONG4):
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            x0 = max(0, int((cx - bw / 2) * width))
            y0 = max(0, int((cy - bh / 2) * height))
            x1 = min(width, int((cx + bw / 2) * width))
            y1 = min(height, int((cy + bh / 2) * height))
            if x1 - x0 < 40 or y1 - y0 < 40:
                continue
            verdict = classify(image[y0:y1, x0:x1], desqueeze)
            if verdict is None or verdict == cls:
                kept += 1
                continue
            parts[0] = str(verdict)
            lines[i] = " ".join(parts)
            changed = True
            fixed += 1
        if changed and args.apply:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checked+kept: {kept}, fixed: {fixed}, apply={args.apply}")


if __name__ == "__main__":
    main()
