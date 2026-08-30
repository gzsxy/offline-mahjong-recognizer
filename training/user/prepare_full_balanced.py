#!/usr/bin/env python3
"""Oversample flower/season images for a recognition-focused fine-tune."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "training/datasets/mj-full-v1"
DEFAULT_OUTPUT = ROOT / "training/datasets/mj-full-balanced-v1"
RARE_IDS = set(range(34, 42))
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
    parser.add_argument("--duplicates", type=int, default=2)
    args = parser.parse_args()
    if args.duplicates < 0:
        raise SystemExit("--duplicates must be non-negative")
    if not (args.source / "data.yaml").exists():
        raise FileNotFoundError(args.source / "data.yaml")
    if args.out.exists():
        shutil.rmtree(args.out)

    counts = {"base": 0, "extra": 0, "rare": 0}
    for split in ("train", "val"):
        source_images = args.source / "images" / split
        source_labels = args.source / "labels" / split
        for image in sorted(source_images.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = source_labels / f"{image.stem}.txt"
            if not label.exists():
                continue
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            rare = any(int(line.split()[0]) in RARE_IDS for line in lines)
            link_or_copy(image, args.out / "images" / split / image.name)
            link_or_copy(label, args.out / "labels" / split / label.name)
            counts["base"] += 1
            if rare:
                counts["rare"] += 1
            if split == "train" and rare:
                for duplicate in range(args.duplicates):
                    prefix = f"rare{duplicate + 1}_"
                    link_or_copy(
                        image,
                        args.out / "images" / split / f"{prefix}{image.name}",
                    )
                    link_or_copy(
                        label,
                        args.out / "labels" / split / f"{prefix}{label.name}",
                    )
                    counts["extra"] += 1

    data = yaml.safe_load((args.source / "data.yaml").read_text())
    data["path"] = str(args.out.resolve())
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"output: {args.out.resolve()}")
    print(f"base images: {counts['base']}, rare images: {counts['rare']}, extras: {counts['extra']}")


if __name__ == "__main__":
    main()
