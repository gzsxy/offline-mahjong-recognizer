#!/usr/bin/env python3
"""构建 v2 的 1280 混合数据集（符号链接合并，复用 build_1280_mix.build_mix）。

- mj-face-1280-mix-v2 = mj-synth-1280-v2（新场景 16k）+ mj-synth-1280-v1（2.5k）
                       + mj-user-face-v1（12）+ mj-v2 随机 30% 回放（seed 44）
- mj-back-1280-mix-v2 = mj-back-synth-1280-v2（新场景 6k）+ mj-back-synth-1280-v1（1.5k）
                       + mj-user-back-1280-v1（24）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_1280_mix import DATASETS, build_mix  # noqa: E402

FACE_MIX = DATASETS / "mj-face-1280-mix-v2"
FACE_SOURCES = [
    (DATASETS / "mj-synth-1280-v2", None),
    (DATASETS / "mj-synth-1280-v1", None),
    (DATASETS / "mj-user-face-v1", None),
    (DATASETS / "mj-v2", 0.30),
]
BACK_MIX = DATASETS / "mj-back-1280-mix-v2"
BACK_SOURCES = [
    (DATASETS / "mj-back-synth-1280-v2", None),
    (DATASETS / "mj-back-synth-1280-v1", None),
    (DATASETS / "mj-user-back-1280-v1", None),
]

if __name__ == "__main__":
    build_mix(FACE_MIX, FACE_SOURCES)
    build_mix(BACK_MIX, BACK_SOURCES)
