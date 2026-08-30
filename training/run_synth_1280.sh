#!/bin/sh
# B 阶段 1280 合成数据生成（CPU，与 MPS 训练并行；顺序执行避免 CPU 争抢）
set -eu
cd "$(dirname "$0")/.."
PY=training/venv/bin/python

echo "[$(date '+%F %T')] synth-1280 face start"
"$PY" training/synth/generate.py --canvas 1280 \
  --out training/datasets/mj-synth-1280-v1 \
  --train 2500 --val 400 --tiles-max 24 --seed 43
echo "[$(date '+%F %T')] synth-1280 face done, back-boost start"
"$PY" training/synth/generate.py --canvas 1280 \
  --out training/datasets/mj-back-synth-1280-v1 \
  --train 1500 --val 250 --tiles-max 24 --seed 44 \
  --back-range 0.35,0.6
echo "[$(date '+%F %T')] synth-1280 all done"
