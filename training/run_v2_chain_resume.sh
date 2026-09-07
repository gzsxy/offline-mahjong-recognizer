#!/bin/sh
# v2 训练链（恢复版）：A2 断点续训 → B1 11x-640 预训 → B2 11x-1280 微调 → C 牌背 11m-1280 长程。
# 2026-09-01 原 run_v2_chain.sh 挂在交互会话下被连带杀死（A2 中断于 e29 33%），
# 本脚本为恢复入口；用 setsid nohup 脱离会话启动，避免再次被会话退出杀掉。
# 注意：不要改回用 train_face_11l_1280_v2.py 重跑 A2——exist_ok=True 会清掉已有 28 轮成果。
set -eu
cd "$(dirname "$0")/.."
PY=training/venv/bin/python

echo "[$(date '+%F %T')] A2 resume: face 11l 1280-v2 from e28 last.pt"
"$PY" training/resume_face_11l_1280_v2.py
echo "[$(date '+%F %T')] B1: face 11x 640 pretrain start"
"$PY" training/train_face_11x_640.py
echo "[$(date '+%F %T')] B2: face 11x 1280 finetune start"
"$PY" training/train_face_11x_1280.py
echo "[$(date '+%F %T')] C: back 11m 1280-v2 long start"
"$PY" training/train_back_11m_1280_v2.py
echo "[$(date '+%F %T')] v2 chain (resume) all done"
