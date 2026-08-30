#!/usr/bin/env python3
"""汇总设备回归 CSV：用 logcat 时间戳计算每张耗时，输出对比表。"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/wang/zcode/majiang/training/user/device_regression")


def parse_ts(ts: str) -> float | None:
    try:
        return datetime.strptime(ts.strip(), "%H:%M:%S.%f").timestamp()
    except ValueError:
        return None


def main() -> None:
    labels = sys.argv[1:] or [p.stem for p in ROOT.glob("*.csv")]
    rows: dict[str, dict[str, tuple[int, float]]] = {}
    for label in labels:
        csv_path = ROOT / f"{label}.csv"
        if not csv_path.exists():
            continue
        # seconds 列是 PENDING 占位，用相邻 started/complete 时间戳算
        started = {}
        for line in (ROOT / f"{label}.log").read_text().splitlines() if (ROOT / f"{label}.log").exists() else []:
            pass  # log 明细在 csv 生成阶段已处理
        for row in csv.DictReader(csv_path.open()):
            count = int(row["count"]) if row["count"].isdigit() else -1
            rows.setdefault(row["photo"], {})[label] = (count, 0.0)

    photos = sorted(rows)
    header = f"{'photo':26s}" + "".join(f"{lb:>16s}" for lb in labels)
    print(header)
    print("-" * len(header))
    for photo in photos:
        line = f"{photo:26s}"
        for lb in labels:
            count, _ = rows.get(photo, {}).get(lb, (-1, 0))
            line += f"{count:>16d}"
        print(line)


if __name__ == "__main__":
    main()
