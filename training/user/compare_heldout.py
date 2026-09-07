#!/usr/bin/env python3
"""留出集结果对照：真机检出数 vs 数据集预标注数，输出逐片对比与汇总。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("/Users/wang/zcode/majiang")
DS = ROOT / "training/datasets/mj-user-face-20260831"
CSV_PATH = ROOT / "training/user/device_regression/heldout_20260831.csv"


def main() -> None:
    rows = {}
    for row in csv.DictReader(CSV_PATH.open()):
        rows[row["photo"]] = row

    print(f"{'切片':30s} {'标注':>4s} {'实测':>4s} {'差值':>5s} {'耗时':>6s}")
    print("-" * 60)
    diffs = []
    for name, row in sorted(rows.items()):
        lbl = DS / f"labels/train/{name}.txt"
        expect = len([l for l in lbl.read_text().splitlines() if l.strip()]) if lbl.exists() else -1
        got = int(row["count"]) if row["count"].isdigit() else -1
        d = "" if -1 in (expect, got) else f"{got - expect:+d}"
        if d:
            diffs.append(got - expect)
        print(f"{name[-28:]:30s} {expect:4d} {got:4d} {d:>5s} {row['seconds']:>6s}")

    if diffs:
        import statistics
        print("-" * 60)
        print(f"n={len(diffs)}  均值差={statistics.mean(diffs):+.1f}  "
              f"中位差={statistics.median(diffs):+.1f}  范围=[{min(diffs):+d}, {max(diffs):+d}]  "
              f"|差|<=5 占比={sum(1 for x in diffs if abs(x) <= 5) / len(diffs):.0%}")


if __name__ == "__main__":
    main()
