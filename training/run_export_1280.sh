#!/bin/sh
# B7 导出：两枚 1280 int8 TFLite（CPU 校准）。face 约 414 张、back 约 282 张校准图。
set -eu
cd "$(dirname "$0")/.."
PY=training/venv/bin/python

echo "[$(date '+%F %T')] export face 11l 1280 int8 start"
"$PY" -c "from ultralytics import YOLO; YOLO('training/runs/mj-face-11l-1280/weights/best.pt').export(format='tflite', imgsz=1280, int8=True, data='training/datasets/calib-face-1280/data.yaml', nms=False, device='cpu')"
echo "[$(date '+%F %T')] face done, export back 11m 1280 int8 start"
"$PY" -c "from ultralytics import YOLO; YOLO('training/runs/mj-back-11m-1280/weights/best.pt').export(format='tflite', imgsz=1280, int8=True, data='training/datasets/calib-back-1280/data.yaml', nms=False, device='cpu')"
echo "[$(date '+%F %T')] both exports done"
ls -lh training/runs/mj-face-11l-1280/weights/*int8* training/runs/mj-back-11m-1280/weights/*int8*
