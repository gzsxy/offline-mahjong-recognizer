#!/usr/bin/env python3
"""YOLO 数据集 -> Label Studio 任务 JSON（带预标注）。

用法：
  training/venv/bin/python training/tools/yolo_to_labelstudio.py \
      --dataset training/datasets/mj-user-face-20260831 --out /tmp/ls_tasks.json

生成的任务 data.image 为 /data/local/?d=<相对路径> 形式，配合
LABEL_STUDIO_LOCAL_FILES_ROOT=<数据集根> 环境变量直接服务本地图片。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

NAMES = ([f"wan{i}" for i in range(1, 10)] + [f"tong{i}" for i in range(1, 10)]
         + [f"tiao{i}" for i in range(1, 10)] + ["back"])
BACK_ID = 27


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ds = Path(args.dataset)

    tasks = []
    for split in ("train", "val"):
        for img in sorted((ds / f"images/{split}").glob("*.jpg")):
            lbl = ds / f"labels/{split}" / (img.stem + ".txt")
            if not lbl.exists():
                continue
            w, h = Image.open(img).size
            results = []
            if lbl.read_text().strip():
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if len(p) != 5:
                        continue
                    cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                    results.append({
                        "from_name": "label", "to_name": "image",
                        "type": "rectanglelabels",
                        "original_width": w, "original_height": h,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": (cx - bw / 2) * 100,
                            "y": (cy - bh / 2) * 100,
                            "width": bw * 100,
                            "height": bh * 100,
                            "rectanglelabels": [NAMES[cid]],
                        },
                    })
            rel = f"images/{split}/{img.name}"
            tasks.append({
                "data": {"image": f"/data/local/?d={rel}", "stem": img.stem},
                "predictions": [{"result": results}] if results else [],
            })

    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False))
    n_boxes = sum(len(t["predictions"][0]["result"]) if t["predictions"] else 0 for t in tasks)
    print(f"任务 {len(tasks)} 张，预标注框 {n_boxes} 个 -> {args.out}")


if __name__ == "__main__":
    main()
