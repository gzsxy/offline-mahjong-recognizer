#!/bin/sh
# v2 训练链：A2 正面 11l-1280 长程 → B1 11x-640 预训 → B2 11x-1280 微调 → C 牌背 11m-1280 长程。
# 顺序执行避免显存争用。前置：synth v2 已生成、mix-v2 已构建、yolo11x.pt 已就位。
set -eu
cd "$(dirname "$0")/.."
PY=training/venv/bin/python

echo "[$(date '+%F %T')] A2: face 11l 1280-v2 long start"
"$PY" training/train_face_11l_1280_v2.py
echo "[$(date '+%F %T')] B1: face 11x 640 pretrain start"
"$PY" training/train_face_11x_640.py
echo "[$(date '+%F %T')] B2: face 11x 1280 finetune start"
"$PY" training/train_face_11x_1280.py
echo "[$(date '+%F %T')] C: back 11m 1280-v2 long start"
"$PY" training/train_back_11m_1280_v2.py
echo "[$(date '+%F %T')] v2 chain all done"
