#!/usr/bin/env python3
"""Build a 15-class detector dataset for honors and flower/season tiles."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training/datasets/mj-full-v1"
DEFAULT_OUTPUT = ROOT / "training/datasets/mj-special-v1"
CLASS_NAMES = (
    ["east", "south", "west", "north", "red", "green", "white"]
    + [f"flower{i}" for i in range(1, 5)]
    + [f"season{i}" for i in range(1, 5)]
)
FULL_TO_SPECIAL = {full_id: special_id for special_id, full_id in enumerate(range(27, 42))}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not (args.source / "data.yaml").exists():
        raise FileNotFoundError(args.source / "data.yaml")
    if args.out.exists():
        shutil.rmtree(args.out)

    images = 0
    special_instances = 0
    for split in ("train", "val"):
        source_images = args.source / "images" / split
        source_labels = args.source / "labels" / split
        for image in sorted(source_images.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = source_labels / f"{image.stem}.txt"
            if not label.exists():
                continue
            output_lines = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                full_id = int(parts[0])
                special_id = FULL_TO_SPECIAL.get(full_id)
                if special_id is not None:
                    output_lines.append(" ".join([str(special_id), *parts[1:5]]))
                    special_instances += 1
            link_or_copy(image, args.out / "images" / split / image.name)
            output_label = args.out / "labels" / split / label.name
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text("\n".join(output_lines) + ("\n" if output_lines else ""))
            images += 1

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"output: {args.out.resolve()}")
    print(f"images: {images}, special instances: {special_instances}")


if __name__ == "__main__":
    main()
