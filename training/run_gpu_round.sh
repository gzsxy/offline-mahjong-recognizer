#!/bin/sh
# 1920 轮全流程编排（在 GPU 盒子上执行）：合成数据(缺则本机生成) -> 混合/校准集
#   -> face 11x@640 预训 -> face 11x@1920 微调 -> back 11m@1920 微调 -> int8 TFLite 导出
# 用法：nohup sh training/run_gpu_round.sh > training/runs/gpu_round.log 2>&1 &
set -eu
cd "$(dirname "$0")/.."
PY=training/venv/bin/python

echo "[$(date '+%F %T')] step0 本机化（数据路径/断点参数）"
"$PY" training/setup_portable_env.py

for d in mj-synth-1920-v1 mj-back-synth-1920-v1; do
  if [ ! -d "training/datasets/$d" ]; then
    echo "[$(date '+%F %T')] step1 生成 $d（CPU，数小时）"
    if [ "$d" = "mj-synth-1920-v1" ]; then
      "$PY" training/synth/generate.py --canvas 1920 --out "training/datasets/$d" \
        --train 2500 --val 400 --tiles-max 24 --seed 43
    else
      "$PY" training/synth/generate.py --canvas 1920 --out "training/datasets/$d" \
        --train 1500 --val 250 --tiles-max 24 --seed 44 --back-range 0.35,0.6
    fi
  fi
done

echo "[$(date '+%F %T')] step2 构建混合集与校准集"
"$PY" training/build_1920_assets.py

echo "[$(date '+%F %T')] step3 face 11x@640 预训（最耗时的一段）"
"$PY" training/train_face_11x_640.py

echo "[$(date '+%F %T')] step4 face 11x@1920 微调"
"$PY" training/train_face_11x_1920.py

echo "[$(date '+%F %T')] step5 back 11m@1920 微调"
"$PY" training/train_back_11m_1920.py

echo "[$(date '+%F %T')] step6 导出 int8 TFLite ×2（CPU 校准）"
"$PY" -c "from ultralytics import YOLO; YOLO('training/runs/mj-face-11x-1920/weights/best.pt').export(format='tflite', imgsz=1920, int8=True, data='training/datasets/calib-face-1920/data.yaml', nms=False, device='cpu')"
"$PY" -c "from ultralytics import YOLO; YOLO('training/runs/mj-back-11m-1920/weights/best.pt').export(format='tflite', imgsz=1920, int8=True, data='training/datasets/calib-back-1920/data.yaml', nms=False, device='cpu')"

echo "[$(date '+%F %T')] 全部完成。传回 Mac：training/runs/{mj-face-11x-640,mj-face-11x-1920,mj-back-11m-1920} 与两枚 tflite"
