#!/usr/bin/env python3
"""Remap & merge mahjong detection datasets into a fixed YOLOv8 class system.

Target class system (fixed ids):
  0-8   wan1..wan9
  9-17  tong1..tong9
  18-26 tiao1..tiao9
  27-33 honor tiles and 34-41 flower/season tiles (when ``--full`` is used)
  27/42 back (28-class/full mode respectively)

Sources:
  ma2nf_v5   real photos, 38 classes (Nm/Np/Ns/Nz/back)
  baq4s_v83  real photos, 42 classes (NB/NC/ND + flowers/seasons/winds/dragons)
  ss6ot_v1   real tile-back photos, 2 classes (Mahjong_back / Mahjong_front)
  mj-synth-v1 synthetic, already 28-class, merged as-is (rename with prefix only)
"""
import argparse
import collections
import os
import random
import re
import shutil
import sys

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAINING = os.path.join(PROJECT_ROOT, "training")
OUT = os.path.join(TRAINING, "datasets", "mj-v2")
PREVIEW_DIR = os.path.join(TRAINING, "assets", "preview")

BASE_TARGET_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["back"]
)
FULL_TARGET_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["east", "south", "west", "north", "red", "green", "white"]
    + [f"flower{i}" for i in range(1, 5)]
    + [f"season{i}" for i in range(1, 5)]
    + ["back"]
)
TARGET_NAMES = BASE_TARGET_NAMES
TARGET_ID = {name: i for i, name in enumerate(TARGET_NAMES)}
OUT = os.path.join(TRAINING, "datasets", "mj-v2")
assert len(BASE_TARGET_NAMES) == 28
assert len(FULL_TARGET_NAMES) == 43

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DROP = "DROP"          # drop this annotation line
EXCLUDE = "EXCLUDE"    # exclude the whole image


# ---------------------------------------------------------------------------
# Per-source mapping rules. Each returns a dict {src_class_index: target_id
# (int) | DROP | EXCLUDE}, built dynamically from the source data.yaml names.
# ---------------------------------------------------------------------------

def rules_ma2nf(names):
    """ma2nf_v5: Nm->wanN, Np->tongN, Ns->tiaoN (0m/0p/0s are red fives -> 5),
     back->back, 1z..7z (honor tiles) -> honor classes in full mode."""
    m = {}
    for idx, name in enumerate(names):
        mo = re.fullmatch(r"([0-9])([mps])", name)
        if mo:
            n = int(mo.group(1))
            if n == 0:
                n = 5  # red five counts as plain five
            suit = {"m": "wan", "p": "tong", "s": "tiao"}[mo.group(2)]
            m[idx] = TARGET_ID[f"{suit}{n}"]
        elif name == "back":
            m[idx] = TARGET_ID["back"]
        elif re.fullmatch(r"[1-7]z", name):
            honor = {
                "1z": "east",
                "2z": "south",
                "3z": "west",
                "4z": "north",
                "5z": "white",
                "6z": "green",
                "7z": "red",
            }[name]
            m[idx] = TARGET_ID.get(honor, DROP)
        else:
            raise ValueError(f"ma2nf: unhandled class name {name!r}")
    return m


def rules_baq4s(names):
    """baq4s_v83: NB->tiaoN (Bamboo), NC->wanN (Character), ND->tongN (Dot).
    Flowers/seasons are dropped; winds and dragons are retained in full mode."""
    m = {}
    for idx, name in enumerate(names):
        mo = re.fullmatch(r"([1-9])([BCD])", name)
        if mo:
            n = int(mo.group(1))
            suit = {"B": "tiao", "C": "wan", "D": "tong"}[mo.group(2)]
            m[idx] = TARGET_ID[f"{suit}{n}"]
        elif name in {"EW", "NW", "SW", "WW", "GD", "RD", "WD"}:
            honor = {
                "EW": "east",
                "SW": "south",
                "WW": "west",
                "NW": "north",
                "RD": "red",
                "GD": "green",
                "WD": "white",
            }[name]
            m[idx] = TARGET_ID.get(honor, DROP)
        elif re.fullmatch(r"[1-4][FS]", name):
            prefix = "flower" if name.endswith("F") else "season"
            m[idx] = TARGET_ID.get(f"{prefix}{name[0]}", DROP)
        else:
            raise ValueError(f"baq4s: unhandled class name {name!r}")
    return m


def rules_ss6ot(names):
    """ss6ot_v1: Mahjong_back->back; any image containing Mahjong_front is
    excluded entirely (unannotated front tiles would become false negatives)."""
    m = {}
    for idx, name in enumerate(names):
        if name == "Mahjong_back":
            m[idx] = TARGET_ID["back"]
        elif name == "Mahjong_front":
            m[idx] = EXCLUDE
        else:
            raise ValueError(f"ss6ot: unhandled class name {name!r}")
    return m


def rules_identity(names):
    """mj-synth-v1: already in the 28-class target system."""
    m = {}
    for idx, name in enumerate(names):
        key = str(name)
        if key not in TARGET_ID:
            raise ValueError(f"synth: class {key!r} not in target system")
        m[idx] = TARGET_ID[key]
    return m


# (source_key, dataset_root, rules_fn, [(src_split, dst_split)], filename_prefix)
SOURCES = [
    ("ma2nf", os.path.join(TRAINING, "datasets/raw/ma2nf_v5"), rules_ma2nf,
     [("train", "train"), ("valid", "val")], "ma2nf_"),
    ("bq", os.path.join(TRAINING, "datasets/raw/baq4s_v83"), rules_baq4s,
     [("train", "train"), ("valid", "val")], "bq_"),
    ("ss6", os.path.join(TRAINING, "datasets/raw/ss6ot_v1"), rules_ss6ot,
     [("train", "train")], "ss6_"),
    ("syn", os.path.join(TRAINING, "datasets/mj-synth-v1"), rules_identity,
     [("train", "train"), ("val", "val")], "syn_"),
]


def load_names(root):
    with open(os.path.join(root, "data.yaml")) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return [str(n) for n in names]


def remap_label_file(label_path, mapping):
    """Returns (lines, excluded, dropped, kept_ids).
    lines: list of 'new_id cx cy w h' strings; excluded: whole image excluded."""
    lines, dropped = [], 0
    kept_ids = []
    if not os.path.exists(label_path):
        return lines, False, 0, kept_ids
    with open(label_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            src_id = int(parts[0])
            action = mapping[src_id]
            if action is EXCLUDE:
                return None, True, dropped, kept_ids
            if action is DROP:
                dropped += 1
                continue
            coords = parts[1:5]
            lines.append(f"{action} {' '.join(coords)}")
            kept_ids.append(action)
    return lines, False, dropped, kept_ids


def link_or_copy(src, dst):
    try:
        if os.path.exists(dst):
            os.unlink(dst)
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="keep the seven honor-tile classes instead of dropping them",
    )
    parser.add_argument("--out", default=None, help="output dataset directory")
    args = parser.parse_args()

    global TARGET_NAMES, TARGET_ID, OUT
    if args.full:
        TARGET_NAMES = FULL_TARGET_NAMES
        TARGET_ID = {name: i for i, name in enumerate(TARGET_NAMES)}
        OUT = os.path.join(TRAINING, "datasets", "mj-full-v1")
    if args.out:
        OUT = os.path.abspath(args.out)
    if args.full and os.path.isdir(OUT):
        shutil.rmtree(OUT)

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    # stats[source][target_id] = instance count
    stats = {k: collections.Counter() for k, *_ in SOURCES}
    split_counts = collections.Counter()          # dst split -> images
    src_split_counts = collections.Counter()      # (source, dst split) -> images
    dropped_lines = collections.Counter()         # source -> dropped annot lines
    excluded_images = collections.Counter()       # source -> excluded images
    corrupt_images = []                           # (source, path)
    written = {"train": set(), "val": set()}      # basenames written per split
    preview_candidates = collections.defaultdict(list)  # source -> [(img, lbl, split)]

    for src_key, root, rules_fn, splits, prefix in SOURCES:
        names = load_names(root)
        mapping = rules_fn(names)
        n_drop = sum(1 for v in mapping.values() if v is DROP)
        print(f"[{src_key}] {len(names)} classes -> {len(mapping) - n_drop} mapped, {n_drop} dropped")

        for src_split, dst_split in splits:
            img_dir = os.path.join(root, "images", src_split)
            lbl_dir = os.path.join(root, "labels", src_split)
            if not os.path.isdir(img_dir):  # raw roboflow layout: <root>/<split>/images
                img_dir = os.path.join(root, src_split, "images")
                lbl_dir = os.path.join(root, src_split, "labels")
            if not os.path.isdir(img_dir):
                print(f"  !! missing {img_dir}, skipped")
                continue

            for fn in sorted(os.listdir(img_dir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in IMG_EXTS:
                    continue
                src_img = os.path.join(img_dir, fn)
                src_lbl = os.path.join(lbl_dir, stem + ".txt")

                lines, excluded, ndrop, kept_ids = remap_label_file(src_lbl, mapping)
                dropped_lines[src_key] += ndrop
                if excluded:
                    excluded_images[src_key] += 1
                    continue
                if args.full and ndrop:
                    # A dropped object must not silently become background: that
                    # would train the detector to suppress valid neighboring
                    # tiles in images containing flowers or seasons.
                    excluded_images[src_key] += 1
                    continue

                # verify image opens with PIL
                try:
                    with Image.open(src_img) as im:
                        im.verify()
                except Exception as e:
                    corrupt_images.append((src_key, src_img, str(e)))
                    continue

                new_stem = prefix + stem
                if new_stem in written[dst_split]:
                    raise RuntimeError(f"name clash after prefixing: {new_stem}")
                written[dst_split].add(new_stem)

                dst_img = os.path.join(OUT, "images", dst_split, new_stem + ".jpg")
                dst_lbl = os.path.join(OUT, "labels", dst_split, new_stem + ".txt")
                link_or_copy(src_img, dst_img)
                with open(dst_lbl, "w") as f:
                    if lines:
                        f.write("\n".join(lines) + "\n")

                for tid in kept_ids:
                    stats[src_key][tid] += 1
                split_counts[dst_split] += 1
                src_split_counts[(src_key, dst_split)] += 1
                preview_candidates[src_key].append((dst_img, dst_lbl, dst_split))

    # ---------------- validation ----------------
    problems = []
    for split in ("train", "val"):
        img_stems = {os.path.splitext(f)[0]
                     for f in os.listdir(os.path.join(OUT, "images", split))
                     if os.path.splitext(f)[1].lower() in IMG_EXTS}
        lbl_stems = {os.path.splitext(f)[0]
                     for f in os.listdir(os.path.join(OUT, "labels", split))
                     if f.endswith(".txt")}
        for stem in sorted(img_stems - lbl_stems):
            problems.append(f"{split}: image without label: {stem}")
        for stem in sorted(lbl_stems - img_stems):
            problems.append(f"{split}: orphan label: {stem}")
        for stem in sorted(lbl_stems & img_stems):
            with open(os.path.join(OUT, "labels", split, stem + ".txt")) as f:
                for ln, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    cid = int(raw.split()[0])
                    if not 0 <= cid < len(TARGET_NAMES):
                        problems.append(f"{split}/{stem}.txt:{ln}: id {cid} out of range")

    # ---------------- data.yaml ----------------
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(TARGET_NAMES)}\n")
        f.write("names:\n")
        for i, n in enumerate(TARGET_NAMES):
            f.write(f"  {i}: {n}\n")

    # ---------------- report ----------------
    print("\n===== per-source per-class instance counts =====")
    total = collections.Counter()
    for src_key, *_ in SOURCES:
        c = stats[src_key]
        parts = ", ".join(
            f"{TARGET_NAMES[i]}:{c[i]}"
            for i in range(len(TARGET_NAMES))
            if c[i]
        )
        print(f"[{src_key}] total={sum(c.values())}  {parts}")
        total.update(c)

    print(f"\n====={len(TARGET_NAMES)}-class histogram (all splits) =====")
    grand = sum(total.values())
    for i in range(len(TARGET_NAMES)):
        bar = "#" * (total[i] * 40 // max(1, max(total.values())))
        print(f"{i:2d} {TARGET_NAMES[i]:6s} {total[i]:7d} {bar}")
    back_n = total[TARGET_ID["back"]]
    print(f"\nTOTAL instances: {grand}")
    print(f"back instances: {back_n} ({back_n / grand * 100:.2f}%)")

    print("\n===== image counts =====")
    for split in ("train", "val"):
        print(f"{split}: {split_counts[split]} images")
        for src_key, *_ in SOURCES:
            n = src_split_counts[(src_key, split)]
            if n:
                print(f"    {src_key}: {n}")
    print(f"\ndropped annotation lines: {dict(dropped_lines)}")
    print(f"excluded images (ss6 front-containing): {dict(excluded_images)}")
    if corrupt_images:
        print(f"CORRUPT images skipped ({len(corrupt_images)}):")
        for c in corrupt_images:
            print("   ", c)

    print("\n===== validation =====")
    if problems:
        print(f"FAILED: {len(problems)} problems")
        for p in problems[:50]:
            print("  ", p)
        sys.exit(1)
    print(
        f"OK: all label ids in [0,{len(TARGET_NAMES) - 1}]; "
        "every image has a label; no orphan labels"
    )

    # ---------------- previews ----------------
    draw_plan = [("ma2nf", 2), ("bq", 2), ("ss6", 1), ("syn", 1)]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    palette = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 200, 40)]
    for src_key, count in draw_plan:
        cands = preview_candidates[src_key]
        picks = random.sample(cands, min(count, len(cands)))
        for i, (img_path, lbl_path, _split) in enumerate(picks, 1):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                draw = ImageDraw.Draw(im)
                with open(lbl_path) as f:
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        cid, cx, cy, w, h = raw.split()[:5]
                        cid = int(cid)
                        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                        color = palette[cid % len(palette)]
                        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                        name = TARGET_NAMES[cid]
                        draw.rectangle([x0, y0 - 22, x0 + 9 * len(name) + 8, y0], fill=color)
                        draw.text((x0 + 4, y0 - 21), name, fill=(0, 0, 0), font=font)
                out_p = os.path.join(PREVIEW_DIR, f"v2_{src_key}_{i}.jpg")
                im.save(out_p, quality=90)
                print(f"preview: {out_p}  <- {os.path.basename(img_path)}")

    print(f"\nDONE. Output at {OUT}")


if __name__ == "__main__":
    main()
