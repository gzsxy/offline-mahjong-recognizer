#!/usr/bin/env python3
"""Build a small replay dataset for user-specific fine-tuning."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = ROOT / "training/datasets/mj-v2"
USER_DATASET = ROOT / "training/datasets/mj-user-back-v1"
USER_SYNTH_DATASET = ROOT / "training/datasets/mj-user-back-synth-v1"
DEFAULT_OUTPUT = ROOT / "training/datasets/mj-user-finetune-v1"
CLASS_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["back"]
)


def candidates(dataset: Path, split: str) -> list[tuple[Path, Path, set[int]]]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    result = []
    for image_path in sorted(image_dir.glob("*.*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        classes = {
            int(line.split()[0])
            for line in label_path.read_text().splitlines()
            if line.strip()
        }
        result.append((image_path, label_path, classes))
    return result


def select_stratified(
    items: list[tuple[Path, Path, set[int]]],
    limit: int,
    per_class_target: int,
    rng: random.Random,
) -> list[tuple[Path, Path, set[int]]]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    selected: list[tuple[Path, Path, set[int]]] = []
    selected_paths: set[Path] = set()
    counts = defaultdict(int)

    for class_id in range(len(CLASS_NAMES)):
        for item in shuffled:
            image_path, _, classes = item
            if image_path in selected_paths or class_id not in classes:
                continue
            if counts[class_id] >= per_class_target or len(selected) >= limit:
                break
            selected.append(item)
            selected_paths.add(image_path)
            for item_class in classes:
                counts[item_class] += 1

    for item in shuffled:
        if len(selected) >= limit:
            break
        if item[0] not in selected_paths:
            selected.append(item)
            selected_paths.add(item[0])
    return selected


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.unlink(missing_ok=True)
        destination.hardlink_to(source)
    except (OSError, AttributeError):
        shutil.copy2(source, destination)


def copy_items(
    items: list[tuple[Path, Path, set[int]]],
    image_dir: Path,
    label_dir: Path,
    prefix: str,
) -> None:
    for image_path, label_path, _ in items:
        link_or_copy(image_path, image_dir / f"{prefix}{image_path.name}")
        link_or_copy(label_path, label_dir / f"{prefix}{label_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=400)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.base_limit <= 0 or args.val_limit <= 0:
        raise SystemExit("dataset limits must be positive")

    if args.out.exists():
        shutil.rmtree(args.out)
    train_image_dir = args.out / "images/train"
    val_image_dir = args.out / "images/val"
    train_label_dir = args.out / "labels/train"
    val_label_dir = args.out / "labels/val"
    for directory in (train_image_dir, val_image_dir, train_label_dir, val_label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    base_train = select_stratified(
        candidates(BASE_DATASET, "train"), args.base_limit, 100, rng
    )
    base_val = select_stratified(
        candidates(BASE_DATASET, "val"), args.val_limit, 20, rng
    )
    user_train = candidates(USER_DATASET, "train")
    user_val = candidates(USER_DATASET, "val")
    user_synth_train = candidates(USER_SYNTH_DATASET, "train")
    user_synth_val = candidates(USER_SYNTH_DATASET, "val")
    copy_items(base_train, train_image_dir, train_label_dir, "base_")
    copy_items(base_val, val_image_dir, val_label_dir, "base_")
    copy_items(user_train, train_image_dir, train_label_dir, "user_")
    copy_items(user_val, val_image_dir, val_label_dir, "user_")
    copy_items(user_synth_train, train_image_dir, train_label_dir, "usersynth_")
    copy_items(user_synth_val, val_image_dir, val_label_dir, "usersynth_")

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"output: {args.out.resolve()}")
    print(
        f"base train: {len(base_train)}, user train: {len(user_train)}, "
        f"user synth train: {len(user_synth_train)}"
    )
    print(
        f"base val: {len(base_val)}, user val: {len(user_val)}, "
        f"user synth val: {len(user_synth_val)}"
    )


if __name__ == "__main__":
    main()
