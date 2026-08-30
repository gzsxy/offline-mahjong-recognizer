#!/bin/sh
# B 阶段 640 预训链：正面 11l（支持断点续训）→ 牌背 11m。顺序执行避免显存争用。
# 迁移到新机器后先执行 training/setup_portable_env.py 再运行本脚本。
set -eu
cd "$(dirname "$0")/.."
PY=training/venv/bin/python
FACE_LAST=training/runs/mj-face-11l-640/weights/last.pt

echo "[$(date '+%F %T')] face 11l 640 pretrain start"
if [ -f "$FACE_LAST" ]; then
    echo "found $FACE_LAST -> resume from checkpoint"
    "$PY" training/resume_face_11l_640.py
else
    "$PY" training/train_face_11l_640.py
fi
echo "[$(date '+%F %T')] face 11l done, back 11m 640 pretrain start"
"$PY" training/train_back_11m_640.py
echo "[$(date '+%F %T')] all 640 pretrains done"
