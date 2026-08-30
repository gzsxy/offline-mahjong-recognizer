#!/usr/bin/env python3
"""标签核验工具：把数据集标注框裁出拼成对照表（牌面 + 类别名），供人工核验。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["back"]
)


def _load_font(size: int):
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "training/datasets/mj-user-face-v1"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=str(ROOT / "training/user/label_sheets"))
    parser.add_argument("--cell", type=int, default=110)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--classes", default="", help="逗号分隔类别名过滤，空=全部")
    parser.add_argument("--mark", action="store_true", help="在格子上标注 文件:行号 便于回改")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(max(14, args.cell // 6))
    mark_font = _load_font(max(10, args.cell // 11))
    wanted = {CLASS_NAMES.index(n) for n in args.classes.split(",") if n}

    images = sorted((dataset / f"images/{args.split}").glob("*.jpg"))
    tiles: list[tuple[str, Image.Image, str]] = []
    for img_path in images:
        label_path = dataset / f"labels/{args.split}/{img_path.stem}.txt"
        if not label_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        for line_no, line in enumerate(label_path.read_text().splitlines()):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if wanted and cls not in wanted:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            x0 = max(0, int((cx - bw / 2) * width))
            y0 = max(0, int((cy - bh / 2) * height))
            x1 = min(width, int((cx + bw / 2) * width))
            y1 = min(height, int((cy + bh / 2) * height))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            mark = f"{img_path.stem}:{line_no}" if args.mark else ""
            tiles.append((CLASS_NAMES[cls], image.crop((x0, y0, x1, y1)), mark))

    cell = args.cell
    label_h = cell // 3
    per_sheet = args.cols * 10
    for sheet_index in range(0, len(tiles), per_sheet):
        chunk = tiles[sheet_index: sheet_index + per_sheet]
        rows = math.ceil(len(chunk) / args.cols)
        sheet = Image.new("RGB", (args.cols * cell, rows * (cell + label_h)), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for i, (name, tile, mark) in enumerate(chunk):
            col, row = i % args.cols, i // args.cols
            tile_copy = tile.copy()
            tile_copy.thumbnail((cell, cell))
            x = col * cell + (cell - tile_copy.width) // 2
            sheet.paste(tile_copy, (x, row * (cell + label_h)))
            color = (90, 220, 120) if name != "back" else (220, 200, 90)
            draw.text(
                (col * cell + 4, row * (cell + label_h) + cell + 2),
                name, fill=color, font=font,
            )
            if mark:
                draw.text(
                    (col * cell + 4, row * (cell + label_h) + cell + label_h - 14),
                    mark, fill=(255, 160, 60), font=mark_font,
                )
        out = out_dir / f"sheet_{args.split}_{sheet_index // per_sheet:02d}.jpg"
        sheet.save(out, quality=90)
        print(f"{out}  ({len(chunk)} tiles)")


if __name__ == "__main__":
    main()
