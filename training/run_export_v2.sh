#!/bin/sh
# v2 冠军模型导出：face 11l-1280-v2 int8 + back 11m-1280-v2 int8 + back v2 FP32（A/B 备胎）
set -eu
cd "$(dirname "$0")/.."
PY=training/venv/bin/python

echo "[$(date '+%T')] face v2 int8 start"
"$PY" -c "from ultralytics import YOLO; YOLO('training/runs/mj-face-11l-1280-v2/weights/best.pt').export(format='tflite', imgsz=1280, int8=True, data='training/datasets/calib-face-1280/data.yaml', nms=False, device='cpu')"
echo "[$(date '+%T')] back v2 int8 start"
"$PY" -c "from ultralytics import YOLO; YOLO('training/runs/mj-back-11m-1280-v2/weights/best.pt').export(format='tflite', imgsz=1280, int8=True, data='training/datasets/calib-back-1280/data.yaml', nms=False, device='cpu')"
echo "[$(date '+%T')] back v2 fp32 start"
"$PY" -c "from ultralytics import YOLO; YOLO('training/runs/mj-back-11m-1280-v2/weights/best.pt').export(format='tflite', imgsz=1280, quantize=32, nms=False, device='cpu')"
echo "[$(date '+%T')] all done"
ls -lh training/runs/mj-face-11l-1280-v2/weights/*.tflite training/runs/mj-back-11m-1280-v2/weights/*.tflite
